# TurboQuant KV Cache — Full-Document Long-Context Eval

**val_bpb: TBD** | **TBD MB** | 8×H100 SXM, ~600s training

## Key Innovation: TurboQuant KV Cache at Eval

Current leaderboard eval: sliding window, stride=64, context=2048 tokens — each window re-runs the full forward pass from scratch, so each token sees at most 2048 tokens of preceding context.

This submission: process the validation set in 64-token chunks using a TurboQuant-compressed KV cache. Each token attends to the **full preceding document**, not just 2048 tokens.

**No training changes.** Architecture identical to the `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` record (1.1194 BPB).

### TurboQuant_mse Algorithm (Zandieh et al. 2025, arXiv:2504.19874)

1. Normalize each KV vector to unit sphere: `x̂ = x / ‖x‖`
2. Apply a fixed random orthogonal rotation: `y = x̂ @ R.T`
3. Per-coordinate Lloyd-Max scalar quantization against an N(0, 1/D) codebook
4. Store: `uint8` indices + `fp16` norm scalar per head

| bits | MSE distortion | vs fp16 |
|------|---------------|---------|
| 2    | ~0.117        | 8×      |
| **3** | **~0.030**  | **5.3×** |
| 4    | ~0.009        | 4×      |

At 3 bits, quality is neutral vs fp16 (paper §4.2). Default: `TQ_BITS=3`.

### Why This Helps

The standard sliding eval gives each token a maximum of 2048 tokens of context. With TurboQuant KV caching, each token at position `t` has attended to all `t` preceding tokens. For a 60K-token val set, later tokens now benefit from ~30× more context than before.

## Eval Protocol

```
chunk_size = 64 tokens
warmup_tokens = 128  (first 128 tokens excluded from BPB, same as sliding eval)
bits = 3
seed = 42
```

Process validation set left-to-right in 64-token chunks:
1. `cache.begin_chunk(T_new)` — record current position
2. Forward pass with patched attention (reads/writes TurboQuant cache)
3. `cache.end_chunk()` — advance position counter
4. Compute NLL only on post-warmup tokens

## Architecture (unchanged)

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV heads, GQA) |
| MLP | 3× with LeakyReLU(0.5)² |
| BigramHash | 1536 |
| XSA | Last 4 layers |
| RoPE | Partial (16/64 dims), NTK scaling |
| Weight avg | EMA(0.997) + Tight SWA(every 50) |
| Quantization | GPTQ-lite int6 + lzma |
| Optimizer | Parameter Banking + Parallel Muon |

## Run Command

```bash
TQ_EVAL=1 TQ_BITS=3 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Files

- `train_gpt.py` — base script with TurboQuant eval wired in (3 minimal diffs vs record)
- `turboquant_kv.py` — TurboQuant KV cache module (dependency)

## Credits

- **Base model**: [PR #414](https://github.com/openai/parameter-golf/pull/414) by @signalrush
- **LeakyReLU² activation**: [PR #493](https://github.com/openai/parameter-golf/pull/493) by @parinzee, [PR #518](https://github.com/openai/parameter-golf/pull/518) by @sofiabod
- **Optimizer (Parameter Banking + Parallel Muon)**: [PR #399](https://github.com/openai/parameter-golf/pull/399) by @abaybektursun
- **TTT recipe**: [PR #461](https://github.com/openai/parameter-golf/pull/461) by @Christopher-Lee-McClendon (freeze=0 variant from [PR record](https://github.com/openai/parameter-golf/pull/470))
- **Record base submission (LeakyReLU²+TTT+Muon)**: [2026-03-23](../2026-03-23_LeakyReLU_LegalTTT_ParallelMuon) by @abaybektursun
- **TurboQuant KV cache** (`turboquant_kv.py`): this submission — based on Zandieh et al. 2025, [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
