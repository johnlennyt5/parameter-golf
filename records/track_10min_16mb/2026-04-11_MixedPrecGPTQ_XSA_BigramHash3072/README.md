# Mixed-Precision GPTQ (int5/int6/int7) + XSA-all + BigramHash 3072x112

**val_bpb: 1.1178** (3-seed mean, std 0.0008) | **~15.80 MB** | 8xH100 SXM, 600s | No TTT

**Novel contribution: Hessian-guided mixed-precision GPTQ.** Instead of uniform int6 quantization for all layers, we use per-layer Hessian sensitivity analysis to assign int5 (low sensitivity), int6 (medium), or int7 (high sensitivity) bit widths. This allocates more precision where it matters most. Bit allocation: 19 int5 layers, 33 int6 layers, 14 int7 layers, 3 int8 layers.

Calibration uses AR self-generated data (32 seqs x 2048 tokens, temp=0.8). No val data and no train data are accessed during quantization.

## Results

| Seed | Steps | ms/step | Pre-quant BPB | **Sliding BPB** | Artifact |
|------|-------|---------|---------------|-----------------|----------|
| 314 | 6,925 | 86.65 | 1.1345 | **1.1177** | 15,787,791 |
| 42 | 6,948 | 86.37 | 1.1337 | **1.1170** | 15,816,235 |
| 999 | 6,944 | 86.42 | 1.1344 | **1.1187** | 15,791,603 |
| **Mean** | | | | **1.1178** | |

---

## Main Changes

### 1. Hessian-Guided Mixed-Precision GPTQ (Novel)

Prior submissions use uniform int6 quantization for all quantizable layers. We introduce **per-layer bit-width assignment** based on Hessian sensitivity:

- Compute per-layer Hessian trace from AR self-generated calibration data
- Rank layers by quantization sensitivity (trace of H)
- Bottom 30% sensitivity -> **int5** (saves artifact space)
- Middle 50% -> **int6** (baseline precision)
- Top 20% -> **int7** (preserves critical layers)

This yields bit allocation: `{'int5': 19, 'int6': 33, 'int7': 14, 'int8': 3}`.

The trade-off: int7 layers increase artifact size, requiring more aggressive selective pruning (20-40% of +/-1 values) to fit under 16MB. The net BPB impact vs uniform int6 is approximately neutral on this stack, but the technique demonstrates a principled approach to non-uniform quantization.

### 2. AR Self-Generated Full Hessian GPTQ

Same as PR #1019. After training, the model autoregressively generates 32 sequences of 2048 tokens (temperature=0.8, fixed seed). Hessians H = X^T X are collected from these self-generated sequences. No val data, no train data accessed during quantization.

### 3. Fast Pruning via Linear Interpolation

Replaced the slow binary-search pruning (8+ minutes of repeated LZMA9 compression) with:
- Two exact LZMA9 endpoint measurements (unpruned + fully-pruned)
- Linear interpolation with 10% safety margin
- Post-hoc correction if estimate was slightly off

Pruning phase reduced from ~8 minutes to ~60-90 seconds.

### 4. XSA on all 11 layers, BigramHash 3072 x 112

Same as PR #1019. XSA on all 11 layers (from PR #478 by @gowtham0992). BigramHash 3072 x dim=112.

---

## Architecture

| Component | Setting | First introduced by |
|-----------|---------|---------------------|
| Layers | 11 (512d, 8 GQA heads, 4 KV heads) | Baseline |
| MLP | 3x (1536) with LeakyReLU(0.5)^2 | [#493](https://github.com/openai/parameter-golf/pull/493) @parinzee |
| Attention | XSA on all 11 layers | [#478](https://github.com/openai/parameter-golf/pull/478) @gowtham0992 |
| BigramHash | 3072 x dim=112 | PR #1019 (concept: [#162](https://github.com/openai/parameter-golf/pull/162) @raahilshah) |
| RoPE | Partial (16/64 dims) | [#315](https://github.com/openai/parameter-golf/pull/315) @jfprincz |
| LN Scale | 1/sqrt(layer+1) | [#315](https://github.com/openai/parameter-golf/pull/315) @jfprincz |
| VE128 | Layers 9-10 | [#374](https://github.com/openai/parameter-golf/pull/374) @unnir |
| SmearGate | Position-mixing gate | [#65](https://github.com/openai/parameter-golf/pull/65) @aquariouseworkman |
| U-Net skips | Encoder-decoder connections | [#289](https://github.com/openai/parameter-golf/pull/289) |
| Weight avg | EMA(0.997) + Tight SWA(every 50) | [#401](https://github.com/openai/parameter-golf/pull/401) @newjordan |
| Quantization | **Mixed-Precision GPTQ (int5/int6/int7, AR self-gen)** | **This work** |
| Compression | LZMA preset=9 | [#160](https://github.com/openai/parameter-golf/pull/160) @ChaseWNorton |
| Warmdown | 4000 iterations | [#364](https://github.com/openai/parameter-golf/pull/364) @shikhar1729 |
| Optimizer | Parallel Muon + Parameter Banking | [#399](https://github.com/openai/parameter-golf/pull/399) @abaybektursun |
| Late QAT | STE at LR scale < 0.15 | [#286](https://github.com/openai/parameter-golf/pull/286) @chris-buckley |
| Selective pruning | +/-1 values by reconstruction error | [#609](https://github.com/openai/parameter-golf/pull/609) @saml212 |
| Flash Attention 3 | Hopper warp-specialized kernels | [#122](https://github.com/openai/parameter-golf/pull/122) @mtybadger |

## Requirements

**Flash Attention 3 (Hopper) is required.** The script imports `flash_attn_interface` directly and was run with PyTorch 2.9.1+cu128.

```bash
pip install --break-system-packages flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291
pip install sentencepiece zstandard
python3 -c "from flash_attn_interface import flash_attn_func; import sentencepiece, zstandard; print('deps OK')"
```

## Run Command

```bash
SEED=314 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Lineage

```
PR #1019 (Current SOTA, 1.1147) — AR Self-Gen GPTQ + XSA-all + BigramHash 3072x112
    └── This work adds:
        ├── Mixed-precision GPTQ: Hessian-guided int5/int6/int7 bit allocation (novel)
        ├── Fast pruning: linear interpolation replaces 8-min binary search (~60-90s)
        └── AR calibration reduced: 32 seqs (down from 64)
```
