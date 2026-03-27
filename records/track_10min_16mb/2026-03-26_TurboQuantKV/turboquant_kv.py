from __future__ import annotations
import math
import time
from contextlib import contextmanager
from typing import Generator

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel


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
    if T_total > train_seq_len:
        scale = T_total / train_seq_len
        base  = base * (scale ** (rope_dims / (rope_dims - 2)))
    inv_freq = 1.0 / (base ** (torch.arange(0, rope_dims, 2, dtype=torch.float32, device=device) / rope_dims))
    freqs = torch.outer(torch.arange(T_total, device=device, dtype=inv_freq.dtype), inv_freq)
    return freqs.cos()[None, :, None, :].to(dtype), freqs.sin()[None, :, None, :].to(dtype)


class ExtendedKVCache:
    # Full-precision (bf16) KV cache for extended context eval.
    # Pre-allocates [num_layers, max_tokens, H, D] GPU buffer — no quantization.
    # append() writes k,v directly in-place. get() is a free slice.
    # Gives each val token up to max_tokens of preceding context
    # instead of the 2048-token sliding window.

    def __init__(self, num_layers, num_kv_heads, head_dim, device=None, max_tokens=8192):
        self.num_layers   = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim     = head_dim
        self.device       = device or torch.device("cuda")
        self.max_tokens   = max_tokens

        buf_shape = (num_layers, max_tokens, num_kv_heads, head_dim)
        self._k_buf: Tensor = torch.empty(buf_shape, dtype=torch.bfloat16, device=self.device)
        self._v_buf: Tensor = torch.empty(buf_shape, dtype=torch.bfloat16, device=self.device)

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
        pos = self._total_tokens
        T   = k.shape[0]
        self._k_buf[layer, pos:pos + T] = k.to(torch.bfloat16)
        self._v_buf[layer, pos:pos + T] = v.to(torch.bfloat16)

    def get(self, layer: int, dev=None) -> tuple[Tensor, Tensor]:
        tot = self._total_tokens + self._chunk_T
        k = self._k_buf[layer, :tot]
        v = self._v_buf[layer, :tot]
        if dev is not None and dev != self._k_buf.device:
            k, v = k.to(dev), v.to(dev)
        return k, v

    def clear(self) -> None:
        self._total_tokens  = 0
        self._chunk_T       = 0
        self._cos_sin_key   = None
        self._cos_sin_val   = None

    def memory_bytes(self) -> int:
        return (self._k_buf.numel() + self._v_buf.numel()) * 2  # bf16


@contextmanager
def turboquant_attention(model, cache: ExtendedKVCache) -> Generator[None, None, None]:
    # Patches every CausalSelfAttention to use the extended KV cache.
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

            # Cos/sin: all layers in one chunk share the same T_total — compute once
            T_total     = cache.pos_offset + T_new
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

            q_s = q.permute(0, 2, 1, 3)
            k_s = k_full.unsqueeze(0).permute(0, 2, 1, 3).to(q_s)
            v_s = v_full.unsqueeze(0).permute(0, 2, 1, 3).to(q_s)
            reps = H // Hkv
            k_s  = k_s.repeat_interleave(reps, dim=1)
            v_s  = v_s.repeat_interleave(reps, dim=1)

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
                        is_boundary_token_lut, device, bits=None, chunk_size=512,
                        warmup_tokens=128, seed=None, max_context_tokens=8192,
                        rank=0, world_size=1) -> tuple[float, float]:
    model.eval()
    attn0 = model.blocks[0].attn
    cache = ExtendedKVCache(model.num_layers, attn0.num_kv_heads, attn0.head_dim,
                            device=device, max_tokens=max_context_tokens)

    total_tokens = val_tokens.numel() - 1
    if world_size > 1:
        per_rank  = (total_tokens + world_size - 1) // world_size
        tok_start = rank * per_rank
        tok_end   = min(tok_start + per_rank, total_tokens)
        val_tokens    = val_tokens[tok_start : tok_end + 1]
        total_tokens  = tok_end - tok_start
    else:
        tok_start = 0

    loss_sum = token_count = byte_count = torch.zeros((), dtype=torch.float64, device=device)
    tokens_since_clear = 0
    total_chunks = (total_tokens + chunk_size - 1) // chunk_size
    t_last = time.perf_counter()

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

            if chunk_idx < 3 and rank == 0:
                torch.cuda.synchronize()
                t_pre = time.perf_counter()
                cache.begin_chunk(T_new)
                logits = model.forward_logits(x)
                torch.cuda.synchronize()
                t_post_fwd = time.perf_counter()
                cache.end_chunk()
                t_post = time.perf_counter()
                print(f"  [KV timing chunk {chunk_idx}] "
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
                print(f"  extended_kvcache [rank {rank}]: chunk {chunk_idx}/{total_chunks} "
                      f"({100*chunk_idx/total_chunks:.1f}%) — {ms_per_chunk:.2f} ms/chunk", flush=True)
                t_last = t_now

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

    if world_size > 1:
        import torch.distributed as dist
        for t in (loss_sum, token_count, byte_count):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / token_count).item()
    if rank == 0:
        print(f"[ExtendedKV] {cache.seq_len} tokens/rank, "
              f"{cache.memory_bytes()/1024/1024:.1f} MB buffer")
    model.train()
    return val_loss, (val_loss / math.log(2.0)) * (token_count.item() / byte_count.item())
