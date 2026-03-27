from __future__ import annotations
import math
import time
from contextlib import contextmanager
from typing import Generator

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

# Lloyd-Max centroids for N(0,1) scalar quantization, symmetric about 0
_CODEBOOKS: dict[int, list[float]] = {
    1: [-0.7979,  0.7979],
    2: [-1.5104, -0.4528,  0.4528,  1.5104],
    3: [-2.1520, -1.3439, -0.7560, -0.2451,  0.2451,  0.7560,  1.3439,  2.1520],
    4: [-2.7326, -2.0690, -1.6180, -1.2560, -0.9420, -0.6570, -0.3880, -0.1284,
         0.1284,  0.3880,  0.6570,  0.9420,  1.2560,  1.6180,  2.0690,  2.7326],
}


def _apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int = 0) -> Tensor:
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        return torch.cat((torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1), x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


def _build_cos_sin(rope_dims, base, train_seq_len, T_total, device, dtype):
    # NTK scaling when evaluating beyond training length
    if T_total > train_seq_len:
        scale = T_total / train_seq_len
        base  = base * (scale ** (rope_dims / (rope_dims - 2)))
    inv_freq = 1.0 / (base ** (torch.arange(0, rope_dims, 2, dtype=torch.float32, device=device) / rope_dims))
    freqs = torch.outer(torch.arange(T_total, device=device, dtype=inv_freq.dtype), inv_freq)
    return freqs.cos()[None, :, None, :].to(dtype), freqs.sin()[None, :, None, :].to(dtype)


class TurboQuantKVCache:
    # TurboQuant_mse (Zandieh et al. 2025): normalize → random rotate → Lloyd-Max quantize
    # At 3.5 bits quality-neutral vs fp16; at 2.5 bits marginal loss (per paper §4.2)
    #
    # GPU-resident dequantized buffer: pre-allocate [num_layers, max_tokens, H, D] on GPU.
    # append() writes new tokens in-place — no allocation per chunk.
    # get() is a free slice — no concatenation, no decompression.
    # _cos_sin_key/val: shared cos/sin table across all layers within one chunk (1× instead of 11×).

    def __init__(self, num_layers, num_kv_heads, head_dim, bits=3, device=None, seed=42,
                 max_tokens=8192):
        assert bits in _CODEBOOKS
        self.num_layers   = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim     = head_dim
        self.bits         = bits
        self.device       = device or torch.device("cuda")
        self.max_tokens   = max_tokens

        # Scale codebook from N(0,1) to N(0,1/D) — what each coord looks like after rotation
        cb = torch.tensor(_CODEBOOKS[bits], dtype=torch.float32, device=self.device)
        self.codebook: Tensor = cb / math.sqrt(head_dim)

        # One orthogonal rotation per layer, stored on CPU
        rng = torch.Generator()
        rng.manual_seed(seed)
        self._rot_cpu: list[Tensor] = []
        for _ in range(num_layers):
            Q, _ = torch.linalg.qr(torch.randn(head_dim, head_dim, generator=rng))
            self._rot_cpu.append(Q.contiguous())

        # CPU compressed storage for memory_bytes() accounting only
        self._k_idx: list[list[Tensor]] = [[] for _ in range(num_layers)]
        self._v_idx: list[list[Tensor]] = [[] for _ in range(num_layers)]
        self._k_nrm: list[list[Tensor]] = [[] for _ in range(num_layers)]
        self._v_nrm: list[list[Tensor]] = [[] for _ in range(num_layers)]

        # Pre-allocated GPU dequantized buffers — written in-place, never reallocated
        buf_shape = (num_layers, max_tokens, num_kv_heads, head_dim)
        self._k_buf: Tensor = torch.empty(buf_shape, dtype=torch.bfloat16, device=self.device)
        self._v_buf: Tensor = torch.empty(buf_shape, dtype=torch.bfloat16, device=self.device)

        # Cos/sin memo: all layers in a chunk share the same T_total and dtype,
        # so compute once and reuse. Key = (T_total, dtype).
        self._cos_sin_key: tuple | None = None
        self._cos_sin_val: tuple[Tensor, Tensor] | None = None

        self._total_tokens: int = 0
        self._chunk_T:      int = 0

    def begin_chunk(self, T_new: int) -> None:
        self._chunk_T = T_new

    def end_chunk(self) -> None:
        self._total_tokens += self._chunk_T
        self._chunk_T = 0

    @property
    def pos_offset(self) -> int:
        return self._total_tokens

    @property
    def seq_len(self) -> int:
        return self._total_tokens

    def append(self, layer: int, k: Tensor, v: Tensor) -> None:
        rot = self._rot_cpu[layer].to(k.device, k.dtype)
        T, H, D = k.shape
        k_idx, k_nrm = self._quant(k.reshape(T * H, D), rot)
        v_idx, v_nrm = self._quant(v.reshape(T * H, D), rot)
        # CPU compressed storage for memory accounting
        self._k_idx[layer].append(k_idx.reshape(T, H, D).cpu())
        self._v_idx[layer].append(v_idx.reshape(T, H, D).cpu())
        self._k_nrm[layer].append(k_nrm.reshape(T, H).half().cpu())
        self._v_nrm[layer].append(v_nrm.reshape(T, H).half().cpu())
        # In-place write into pre-allocated GPU buffer
        pos = self._total_tokens
        self._k_buf[layer, pos:pos + T] = self._dequant(k_idx, k_nrm, rot).reshape(T, H, D).to(torch.bfloat16)
        self._v_buf[layer, pos:pos + T] = self._dequant(v_idx, v_nrm, rot).reshape(T, H, D).to(torch.bfloat16)

    def get(self, layer: int, dev=None) -> tuple[Tensor, Tensor]:
        tot = self._total_tokens + self._chunk_T  # includes current chunk
        k = self._k_buf[layer, :tot]
        v = self._v_buf[layer, :tot]
        if dev is not None and dev != self._k_buf.device:
            k, v = k.to(dev), v.to(dev)
        return k, v

    def clear(self) -> None:
        for l in range(self.num_layers):
            self._k_idx[l].clear(); self._v_idx[l].clear()
            self._k_nrm[l].clear(); self._v_nrm[l].clear()
        self._total_tokens = 0
        self._chunk_T = 0
        self._cos_sin_key = None
        self._cos_sin_val = None

    def memory_bytes(self) -> int:
        total = 0
        for l in range(self.num_layers):
            for t in self._k_idx[l]: total += t.numel()
            for t in self._v_idx[l]: total += t.numel()
            for t in self._k_nrm[l]: total += t.numel() * 2
            for t in self._v_nrm[l]: total += t.numel() * 2
        return total

    def _quant(self, x: Tensor, rot: Tensor) -> tuple[Tensor, Tensor]:
        norms  = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        y      = (x / norms) @ rot.T
        idx    = (y.unsqueeze(-1) - self.codebook).abs().argmin(dim=-1).to(torch.uint8)
        return idx, norms.squeeze(-1).float()

    def _dequant(self, idx: Tensor, norms: Tensor, rot: Tensor) -> Tensor:
        return (self.codebook[idx.long()] @ rot) * norms.float().unsqueeze(-1)


@contextmanager
def turboquant_attention(model, cache: TurboQuantKVCache) -> Generator[None, None, None]:
    # Patches every CausalSelfAttention to use the KV cache.
    # Enter ONCE before the chunk loop — not once per chunk.
    orig: dict[int, object] = {}

    def _make_patched(layer_idx: int, attn):
        orig[layer_idx] = attn.forward

        def _patched(x, q_w, k_w, v_w, out_w, v_embed=None, v0=None):
            bsz, T_new, dim = x.shape
            H, Hkv, Dh = attn.num_heads, attn.num_kv_heads, attn.head_dim

            q = F.linear(x, q_w.to(x.dtype)).reshape(bsz, T_new, H,   Dh)
            k = F.linear(x, k_w.to(x.dtype)).reshape(bsz, T_new, Hkv, Dh)
            v = F.linear(x, v_w.to(x.dtype))
            if v_embed is not None: v = v + v_embed
            v = v.reshape(bsz, T_new, Hkv, Dh)

            raw_v = v if attn.value_residual else None
            if attn.value_residual and v0 is not None:
                lam = attn.vr_lambda.to(dtype=v.dtype)
                v   = lam[0] * v0 + lam[1] * v

            q = F.rms_norm(q, (Dh,))
            k = F.rms_norm(k, (Dh,))

            # Cos/sin: all layers in one chunk share the same T_total+dtype — compute once
            T_total = cache.pos_offset + T_new
            cos_sin_key = (T_total, q.dtype)
            if cache._cos_sin_key != cos_sin_key:
                cos, sin = _build_cos_sin(attn.rope_dims, attn.rotary.base,
                                          attn.rotary.train_seq_len,
                                          T_total, x.device, q.dtype)
                cache._cos_sin_key = cos_sin_key
                cache._cos_sin_val = (cos, sin)
            cos, sin = cache._cos_sin_val

            cos_new = cos[:, cache.pos_offset:, :, :]
            sin_new = sin[:, cache.pos_offset:, :, :]
            q = _apply_rotary_emb(q, cos_new, sin_new, attn.rope_dims)
            k = _apply_rotary_emb(k, cos_new, sin_new, attn.rope_dims)
            q = q * attn.q_gain.to(dtype=q.dtype)[None, None, :, None]

            cache.append(layer_idx, k[0], v[0])
            k_full, v_full = cache.get(layer_idx, dev=x.device)
            T_cache = k_full.size(0)

            # SDPA layout: [bsz, nheads, seqlen, head_dim]
            q_s = q.permute(0, 2, 1, 3)
            k_s = k_full.unsqueeze(0).permute(0, 2, 1, 3).to(q_s)
            v_s = v_full.unsqueeze(0).permute(0, 2, 1, 3).to(q_s)
            reps = H // Hkv
            k_s  = k_s.repeat_interleave(reps, dim=1)
            v_s  = v_s.repeat_interleave(reps, dim=1)

            # Bool mask: True=attend. All cached tokens visible; causal only within new chunk
            T_prev    = T_cache - T_new
            attn_mask = torch.ones(1, 1, T_new, T_cache, device=x.device, dtype=torch.bool)
            if T_new > 1:
                attn_mask[0, 0, :, T_prev:] = torch.tril(
                    torch.ones(T_new, T_new, device=x.device, dtype=torch.bool)
                )

            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                y = F.scaled_dot_product_attention(q_s, k_s, v_s, attn_mask=attn_mask)

            if attn.use_xsa:
                v_dq = v_full[-T_new:].unsqueeze(0).to(q_s)
                y = attn._xsa_efficient(y.permute(0, 2, 1, 3), v_dq).permute(0, 2, 1, 3)
            if attn.gated_attention:
                y = y.permute(0, 2, 1, 3) * torch.sigmoid(attn.attn_gate(x)).unsqueeze(-1)
                y = y.permute(0, 2, 1, 3)

            return F.linear(y.permute(0, 2, 1, 3).reshape(bsz, T_new, dim), out_w.to(x.dtype)), raw_v

        return _patched

    for i, block in enumerate(model.blocks):
        block.attn.forward = _make_patched(i, block.attn)
    try:
        yield
    finally:
        for i, block in enumerate(model.blocks):
            block.attn.forward = orig[i]


def eval_val_turboquant(model, val_tokens, base_bytes_lut, has_leading_space_lut,
                        is_boundary_token_lut, device, bits=3, chunk_size=512,
                        warmup_tokens=128, seed=42, max_context_tokens=8192,
                        rank=0, world_size=1) -> tuple[float, float]:
    model.eval()
    attn0 = model.blocks[0].attn
    cache = TurboQuantKVCache(model.num_layers, attn0.num_kv_heads, attn0.head_dim,
                              bits=bits, device=device, seed=seed,
                              max_tokens=max_context_tokens)

    # Shard val_tokens across ranks so all GPUs eval in parallel
    total_tokens = val_tokens.numel() - 1
    if world_size > 1:
        per_rank  = (total_tokens + world_size - 1) // world_size
        tok_start = rank * per_rank
        tok_end   = min(tok_start + per_rank, total_tokens)
        val_tokens = val_tokens[tok_start : tok_end + 1]  # +1 for targets
        total_tokens = tok_end - tok_start
    else:
        tok_start = 0

    loss_sum = token_count = byte_count = torch.zeros((), dtype=torch.float64, device=device)
    tokens_since_clear = 0
    total_chunks = (total_tokens + chunk_size - 1) // chunk_size
    t_last = time.perf_counter()

    # Patch attention once for the whole eval — not once per chunk
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16), \
         turboquant_attention(model, cache):
        for chunk_idx, chunk_start in enumerate(range(0, total_tokens, chunk_size)):
            if tokens_since_clear >= max_context_tokens:
                cache.clear()
                tokens_since_clear = 0

            chunk_end = min(chunk_start + chunk_size, total_tokens)
            T_new = chunk_end - chunk_start
            x = val_tokens[chunk_start    : chunk_end    ].to(device).long().unsqueeze(0)
            y = val_tokens[chunk_start + 1: chunk_end + 1].to(device).long()

            # First 3 chunks: print GPU-synced breakdown to show where time goes
            if chunk_idx < 3 and rank == 0:
                torch.cuda.synchronize()
                t_pre = time.perf_counter()
                cache.begin_chunk(T_new)
                logits = model.forward_logits(x)
                torch.cuda.synchronize()
                t_post_fwd = time.perf_counter()
                cache.end_chunk()
                t_post = time.perf_counter()
                print(f"  [TQ timing chunk {chunk_idx}] "
                      f"fwd={1000*(t_post_fwd-t_pre):.1f}ms  "
                      f"total={1000*(t_post-t_pre):.1f}ms  "
                      f"T_new={T_new} T_cache={cache.seq_len}", flush=True)
            else:
                cache.begin_chunk(T_new)
                logits = model.forward_logits(x)
                cache.end_chunk()
            tokens_since_clear += T_new

            if chunk_idx % 1000 == 0 and chunk_idx > 0:
                t_now = time.perf_counter()
                ms_per_chunk = (t_now - t_last) / 1000 * 1000
                print(f"  turboquant_kv [rank {rank}]: chunk {chunk_idx}/{total_chunks} "
                      f"({100*chunk_idx/total_chunks:.1f}%) — {ms_per_chunk:.2f} ms/chunk", flush=True)
                t_last = t_now

            # Absolute position of first token in this chunk (accounts for rank sharding)
            abs_chunk_start = tok_start + chunk_start
            s = max(0, warmup_tokens - abs_chunk_start)
            if s >= T_new:
                continue

            nll = F.cross_entropy(logits[0, s:].float(), y[s:], reduction="none").to(torch.float64)
            loss_sum    = loss_sum    + nll.sum()
            token_count = token_count + float(T_new - s)
            tgt  = y[s:]
            prev = x[0, s:]
            tb   = base_bytes_lut[tgt].to(torch.float64)
            tb  += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
            byte_count = byte_count + tb.sum()

    # All-reduce across ranks when running distributed
    if world_size > 1:
        import torch.distributed as dist
        for t in (loss_sum, token_count, byte_count):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / token_count).item()
    if rank == 0:
        print(f"[TurboQuant] {cache.seq_len} tokens/rank, {cache.memory_bytes()/1024/1024:.1f} MB cache")
    model.train()
    return val_loss, (val_loss / math.log(2.0)) * (token_count.item() / byte_count.item())
