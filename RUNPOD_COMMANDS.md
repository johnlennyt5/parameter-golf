# RunPod Commands for Priority 1 Testing

## ✅ Code Status

**Branch:** `sota-integration-1.06`
**GitHub:** https://github.com/johnlennyt5/parameter-golf/tree/sota-integration-1.06
**Status:** ✅ Syntax validated, all components present, ready to run

**Changes Pushed:**
- Cross-layer parameter sharing (79% param reduction)
- Kronecker-Factored Error Correction (KFEC)
- Fixed ALS optimization algorithm
- Added verification test

---

## Setup Commands (RunPod)

### 1. Clone and Checkout Branch

```bash
cd ~
git clone https://github.com/johnlennyt5/parameter-golf.git
cd parameter-golf
git checkout sota-integration-1.06
git pull origin sota-integration-1.06
```

### 2. Install Dependencies

```bash
# Install base dependencies
pip install sentencepiece torch numpy

# Install FlashAttention 3 (critical!)
pip install --no-deps flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

# Install lrzip for compression (if not already installed)
sudo apt-get update && sudo apt-get install -y lrzip
```

### 3. Download Dataset

```bash
# Set environment variables
export DATA_PATH=./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved
export TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
export VOCAB_SIZE=8192

# Download dataset (sp8192 variant)
python3 data/cached_challenge_fineweb.py --variant sp8192
```

---

## Test Commands

### Option 1: Quick Validation (Single Seed - RECOMMENDED FIRST)

This runs a **single seed** to verify the code works before using credits on 3-seed run:

```bash
cd ~/parameter-golf

# Priority 1: Cross-Layer Sharing + KFEC
DATA_PATH=./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
VOCAB_SIZE=8192 \
CASEOPS_ENABLED=1 \
SEED=42 \
ITERATIONS=20000 \
MAX_WALLCLOCK_SECONDS=600 \
CROSS_LAYER_SHARING=1 \
DELTA_RANK=32 \
KFEC_ENABLED=1 \
KFEC_RANK=8 \
KFEC_TOP_K=3 \
KFEC_FACTOR_BITS=4 \
KFEC_ALS_ITERS=10 \
LQER_ENABLED=0 \
PHASED_TTT_ENABLED=1 \
PHASED_TTT_PREFIX_DOCS=2500 \
PHASED_TTT_NUM_PHASES=3 \
EMBED_BITS=7 \
MATRIX_LR=0.026 \
MIN_LR=0.1 \
MLP_CLIP_SIGMAS=11.5 \
ATTN_CLIP_SIGMAS=13.0 \
EMBED_CLIP_SIGMAS=14.0 \
GRAD_CLIP_NORM=0.3 \
TTT_CHUNK_SIZE=48 \
WARMUP_STEPS=20 \
MUON_BACKEND_STEPS=5 \
GLOBAL_TTT_MOMENTUM=0.9 \
WARMDOWN_FRAC=0.85 \
BETA2=0.99 \
TTT_BETA2=0.99 \
TTT_WEIGHT_DECAY=0.5 \
TTT_LORA_RANK=80 \
SPARSE_ATTN_GATE_SCALE=0.5 \
GPTQ_RESERVE_SECONDS=0.5 \
GPTQ_CALIBRATION_BATCHES=16 \
VAL_LOSS_EVERY=0 \
GATED_ATTN_QUANT_GATE=1 \
SPARSE_ATTN_GATE_ENABLED=1 \
GATE_WINDOW=12 \
SMEAR_GATE_ENABLED=1 \
FUSED_CE_ENABLED=1 \
COMPRESSOR=pergroup \
NCCL_NET=Socket \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

**Expected output:**
- Training should complete in ~600s
- Artifact size: **13-14 MB** (vs SOTA 15.9 MB)
- val_bpb: **1.036-1.055** (target: beat SOTA 1.0611)

---

### Option 2: Full 3-Seed Run (After Validation)

Only run this if seed=42 validation succeeds:

```bash
cd ~/parameter-golf

for SEED in 0 42 1234; do
    echo "========================================"
    echo "Running seed $SEED"
    echo "========================================"

    DATA_PATH=./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
    TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
    VOCAB_SIZE=8192 \
    CASEOPS_ENABLED=1 \
    SEED=$SEED \
    ITERATIONS=20000 \
    MAX_WALLCLOCK_SECONDS=600 \
    CROSS_LAYER_SHARING=1 \
    DELTA_RANK=32 \
    KFEC_ENABLED=1 \
    KFEC_RANK=8 \
    KFEC_TOP_K=3 \
    KFEC_FACTOR_BITS=4 \
    KFEC_ALS_ITERS=10 \
    LQER_ENABLED=0 \
    PHASED_TTT_ENABLED=1 \
    PHASED_TTT_PREFIX_DOCS=2500 \
    PHASED_TTT_NUM_PHASES=3 \
    EMBED_BITS=7 \
    MATRIX_LR=0.026 \
    MIN_LR=0.1 \
    MLP_CLIP_SIGMAS=11.5 \
    ATTN_CLIP_SIGMAS=13.0 \
    EMBED_CLIP_SIGMAS=14.0 \
    GRAD_CLIP_NORM=0.3 \
    TTT_CHUNK_SIZE=48 \
    WARMUP_STEPS=20 \
    MUON_BACKEND_STEPS=5 \
    GLOBAL_TTT_MOMENTUM=0.9 \
    WARMDOWN_FRAC=0.85 \
    BETA2=0.99 \
    TTT_BETA2=0.99 \
    TTT_WEIGHT_DECAY=0.5 \
    TTT_LORA_RANK=80 \
    SPARSE_ATTN_GATE_SCALE=0.5 \
    GPTQ_RESERVE_SECONDS=0.5 \
    GPTQ_CALIBRATION_BATCHES=16 \
    VAL_LOSS_EVERY=0 \
    GATED_ATTN_QUANT_GATE=1 \
    SPARSE_ATTN_GATE_ENABLED=1 \
    GATE_WINDOW=12 \
    SMEAR_GATE_ENABLED=1 \
    FUSED_CE_ENABLED=1 \
    COMPRESSOR=pergroup \
    NCCL_NET=Socket \
    torchrun --standalone --nproc_per_node=8 train_gpt.py

    echo "Seed $SEED complete"
    echo ""
done
```

---

## What to Look For

### ✅ Success Indicators

**During Training:**
```
Initialized GPT model with cross-layer sharing (delta_rank=32)
Parameter count: ~X.X M (should be < 10M with sharing)
Using KFEC error correction (rank=8, top_k=3)
...
[Step XXXX] loss: X.XXX
GPTQ quantization starting...
Applying KFEC to top-3 error tensors
...
Serializing with per-group compression...
Final artifact: XX.X MB (should be 13-14 MB)
```

**Final Output:**
```
val_bpb: 1.0XXX (target: < 1.055)
Artifact size: XX,XXX,XXX bytes (target: < 15,000,000)
Total time: ~600s
```

### ❌ Failure Indicators

- `ModuleNotFoundError: flash_attn_interface` → FlashAttention not installed
- `FileNotFoundError: data/datasets/...` → Dataset not downloaded
- `RuntimeError: CUDA out of memory` → Reduce batch size or delta_rank
- Artifact size > 16 MB → KFEC not compressing enough
- val_bpb > 1.10 → Performance worse than expected

---

## Debugging

### If training fails:

1. **Check CUDA/GPU:**
   ```bash
   nvidia-smi
   ```

2. **Check dataset:**
   ```bash
   ls -lh data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/
   ```

3. **Check dependencies:**
   ```bash
   python3 -c "import torch; print(torch.__version__)"
   python3 -c "from flash_attn_interface import flash_attn_func"
   python3 -c "import sentencepiece"
   ```

4. **Check lrzip:**
   ```bash
   which lrzip
   lrzip --version
   ```

5. **Run with verbose logging:**
   Add `VERBOSE=1` to the environment variables

---

## Baseline Comparison

Compare your results to SOTA baseline:

| Metric | SOTA (PR #1797) | Priority 1 Target | Your Result |
|--------|-----------------|-------------------|-------------|
| **val_bpb (mean)** | 1.06108 | 1.036-1.041 | ___ |
| **val_bpb (std)** | 0.00090 | < 0.001 | ___ |
| **Artifact size** | 15.9 MB | 13.5-14.5 MB | ___ MB |
| **Training time** | ~600s | ~600s | ___ s |

---

## Next Steps Based on Results

### If val_bpb < 1.045:
🎉 **SUCCESS!** We beat SOTA significantly!
→ Submit immediately OR stack Priority 2/3 for even better score

### If val_bpb 1.045-1.055:
✅ **Good progress** - close to SOTA
→ Implement Priority 2 (MoE + Ensemble TTT) to push lower

### If val_bpb 1.055-1.065:
⚠️ **Marginal improvement**
→ May need hyperparameter tuning or Priority 2/3 to beat SOTA

### If val_bpb > 1.065:
❌ **Underperforming**
→ Debug: check logs for errors, verify KFEC is applied, check compression

---

## Key Configuration Explained

**Cross-Layer Sharing:**
- `CROSS_LAYER_SHARING=1` - Enable shared base weights
- `DELTA_RANK=32` - Rank of per-layer deltas (higher = more capacity, less compression)

**KFEC:**
- `KFEC_ENABLED=1` - Use Kronecker factorization instead of LQER
- `LQER_ENABLED=0` - Disable LQER (mutually exclusive)
- `KFEC_RANK=8` - Kronecker rank (lower = more compression, less accuracy)
- `KFEC_TOP_K=3` - Apply to top-3 highest-error tensors
- `KFEC_FACTOR_BITS=4` - Quantization bits for A,B,C factors
- `KFEC_ALS_ITERS=10` - ALS optimization iterations (more = better but slower)

**Other (from SOTA baseline):**
- All other settings copied from PR #1797 (current SOTA)
- `COMPRESSOR=pergroup` - Per-group lrzip compression
- `PHASED_TTT_*` - 3-phase TTT settings
- `SPARSE_ATTN_GATE_*` - Sparse attention gating

---

## Support

If you encounter issues, share:
1. Full error message
2. Last 50 lines of output
3. Artifact size (if training completed)
4. GPU memory usage (`nvidia-smi` output)

Good luck! 🚀
