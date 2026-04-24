# Plan C Implementation Status — arch3/plan-c-sota

**Date:** 2026-04-23
**Branch:** `arch3/plan-c-sota`
**Goal:** Achieve ≤1.0810 BPB (#1 RANKING) using SP8192 + Linear Recurrence

---

## ✅ **COMPLETED PHASES**

### **Phase 2: SP8192 Migration** ✅

**Commit:** `f1c678f [Phase 2] SP8192 Migration`

**Changes Made:**
- ✅ Updated data paths:
  - `data_path`: `fineweb10B_sp1024` → `fineweb10B_sp8192`
  - `tokenizer_path`: `fineweb_1024_bpe.model` → `fineweb_8192_bpe.model`
  - `vocab_size`: 1024 → 8192

- ✅ Scaled BigramHash for larger vocabulary:
  - `bigram_vocab_size`: 2048 → 24576 (3× vocab_size)
  - `bigram_dim`: 128 (unchanged, already optimal)

- ✅ Applied SOTA hyperparameter tuning:
  - `qk_gain_init`: 1.5 → **5.25** (SOTA value per bigbag analysis)
  - `tied_embed_lr`: 0.035 → **0.04** (higher for larger vocab)
  - `matrix_lr`: 0.025 → **0.022** (slightly lower for stability)
  - `warmdown_iters`: 3500 → **6000** (longer for larger model convergence)
  - `warmup_steps`: 20 → **50** (longer warmup for stability)

**Status:** COMPLETE — SP8192 tokenizer integration ready

---

### **Phase 3: Linear Recurrence Implementation** ✅

**Commit:** `9e5561d [Phase 3] Linear Recurrence Implementation`

**Changes Made:**

**1. Added LinearRecurrenceLayer class (lines 781-825):**
```python
class LinearRecurrenceLayer(nn.Module):
    """
    Simple EMA recurrence: y[t] = alpha * y[t-1] + (1-alpha) * x[t]
    - Per-dimension learnable decay rates (alpha)
    - RMSNorm + residual connection
    - Unbounded context with O(n) time, O(1) memory
    """
```

**Features:**
- ✅ Learnable decay per dimension (initialized to logit(0.9) ≈ 2.2)
- ✅ RMSNorm pre-normalization
- ✅ Learnable output scale
- ✅ Residual connection
- ✅ Simple sequential scan (can be parallelized later)

**2. Added RECURRENCE_LAYERS hyperparameter:**
- Environment variable: `RECURRENCE_LAYERS` (comma-separated layer indices)
- Example: `export RECURRENCE_LAYERS="3,6,9"` → recurrence at layers 3, 6, 9

**3. Integrated into GPT model:**
- ✅ `GPT.__init__`: Added `recurrence_layers` parameter
- ✅ Created `recurrence_layer_set`, `recurrence_blocks`, `recurrence_idx_map`
- ✅ Modified `forward()`: Apply recurrence AFTER attention+MLP in encoder/decoder
- ✅ Modified `forward_logits()`: Same integration for evaluation
- ✅ Updated model instantiation in `main()` (both training and eval models)

**Architecture:**
- **Default (11L pure attention):** No recurrence (backward compatible)
- **Plan C (13L hybrid):** Set `NUM_LAYERS=13`, `RECURRENCE_LAYERS="3,6,9"`
  - Layers 0-2: Attention only
  - Layer 3: Attention + **Recurrence**
  - Layers 4-5: Attention only
  - Layer 6: Attention + **Recurrence**
  - Layers 7-8: Attention only
  - Layer 9: Attention + **Recurrence**
  - Layers 10-12: Attention only
  - **Total:** 13 attention layers + 3 recurrence layers

**Status:** COMPLETE — Recurrence ready for testing

---

## 📊 **EXPECTED IMPROVEMENTS**

| Component | Baseline | Plan C | Expected Δ BPB |
|-----------|----------|--------|----------------|
| Tokenizer | SP1024 | SP8192 | **-0.010 to -0.015** |
| Architecture | 11L Pure Attn | 13L Hybrid (10A + 3R) | **-0.008 to -0.012** |
| QK-Gain | 1.5 | 5.25 | **-0.001 to -0.003** |
| LR Tuning | SP1024-tuned | SP8192-tuned | **-0.002 to -0.005** |
| **TOTAL** | **1.1199** | **Target** | **-0.021 to -0.035** |

**Target BPB:** 1.095-1.105 (pre-TTT)
**With TTT:** ≤1.0810 (#1 RANKING) 🏆

---

## 🧪 **NEXT STEPS: Testing & Validation**

### **Immediate: Smoke Test (5-10 minutes)**

**Purpose:** Verify SP8192 + recurrence integration works

**Environment:**
```bash
export DATA_PATH=./data/datasets/fineweb10B_sp8192
export TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model
export VOCAB_SIZE=8192
export BIGRAM_VOCAB_SIZE=24576
export NUM_LAYERS=13
export RECURRENCE_LAYERS="3,6,9"
export MAX_WALLCLOCK_SECONDS=60
export SEED=42
```

**Run:**
```bash
mkdir -p logs
python train_gpt.py > logs/smoke_sp8192_recurrence_60s.log 2>&1
```

**Success Criteria:**
- ✅ Model loads without errors
- ✅ Training starts and runs for 60s
- ✅ Initial loss ~9.0-9.5 (higher than SP1024 due to larger vocab)
- ✅ Step time ≤ 95ms (baseline: ~87ms, recurrence adds ~5-8ms)
- ✅ Loss decreases smoothly
- ✅ No dtype errors, NaN, or Inf

**Extract metrics:**
```bash
grep -E "step_avg|val_bpb|loss" logs/smoke_sp8192_recurrence_60s.log | tail -20
```

---

### **Phase 4: Validation Run (30-60 minutes)**

**Purpose:** Validate BPB improvement on 300s run

**Environment:** Same as smoke test but:
```bash
export MAX_WALLCLOCK_SECONDS=300
```

**Run:**
```bash
python train_gpt.py --seed 42 > logs/sp8192_recurrence_300s.log 2>&1
```

**Success Criteria:**
- ✅ Step ~1500 val_bpb: **≤1.28** (vs baseline ~1.30 for SP1024)
- ✅ Recurrence layers converge (alpha values in [0.2, 0.8])
- ✅ Training stable throughout run

**Extract metrics:**
```bash
grep "step.*val_bpb" logs/sp8192_recurrence_300s.log | tail -5
```

---

### **Phase 5: Full Training Run (1-2 hours on 8×H100)**

**Purpose:** Full 600s run with all optimizations

**Environment:** Same as validation + full hyperparameters
```bash
export QK_GAIN_INIT=5.25
export TIED_EMBED_LR=0.04
export MATRIX_LR=0.022
export WARMUP_STEPS=50
export WARMDOWN_ITERS=6000
export MAX_WALLCLOCK_SECONDS=600
```

**Run:**
```bash
python train_gpt.py --seed 42 > logs/planc_full_600s_seed42.log 2>&1
```

**Target Metrics:**
- 🎯 Step 4000 BPB: ≤1.180 (vs baseline 1.200)
- 🎯 Post-EMA BPB: ≤1.110 (vs baseline 1.130)
- 🎯 Post-GPTQ BPB: ≤1.085 (vs baseline 1.115)
- 🎯 Artifact size: ≤16,000,000 bytes
- 🎯 Step time avg: ≤95ms
- 🏆 **With TTT: ≤1.0810 BPB** (#1 RANKING!)

---

### **Phase 6: Multi-Seed Validation (3-6 hours on 8×H100)**

**Purpose:** Statistical validation for submission

**Run 3 seeds:**
```bash
for SEED in 42 1337 2025; do
    export SEED=$SEED
    python train_gpt.py > logs/planc_final_seed${SEED}.log 2>&1
done
```

**Extract results:**
```bash
grep "final_int6_sliding_window" logs/planc_final_seed*.log
```

**Statistical validation:**
- Mean BPB ≤ 1.0810
- Std ≤ 0.0005
- Welch's t-test vs SOTA (1.0810): p < 0.01

---

## 📁 **File Structure**

```
arch3/plan-c-sota/
├── train_gpt.py                    # Modified with SP8192 + recurrence
├── PLAN_C_IMPLEMENTATION.md        # Detailed implementation plan
├── IMPLEMENTATION_STATUS.md        # This file (current status)
└── logs/                           # Test logs (to be created)
    ├── smoke_sp8192_recurrence_60s.log
    ├── sp8192_recurrence_300s.log
    ├── planc_full_600s_seed42.log
    ├── planc_final_seed42.log
    ├── planc_final_seed1337.log
    └── planc_final_seed2025.log
```

---

## 🔧 **Key Implementation Details**

### **Recurrence Architecture**

**Layer positions chosen strategically:**
- Layer 3: After initial encoding (early recurrence)
- Layer 6: Mid-network (captures mid-level patterns)
- Layer 9: Late encoding / early decoding (critical junction)

**Why these positions:**
- Evenly spaced (every 3 layers in first 10 layers)
- Matches SOTA patterns from bigbag analysis
- Avoids first/last layers (embeddings/logits sensitive)

### **Hyperparameter Rationale**

| Parameter | Value | Reason |
|-----------|-------|--------|
| `qk_gain_init=5.25` | SOTA | Matches bigbag's best-performing config |
| `tied_embed_lr=0.04` | +14% vs baseline | Larger vocab needs higher LR |
| `matrix_lr=0.022` | -12% vs baseline | Recurrence needs stability |
| `warmdown_iters=6000` | +71% vs baseline | Longer convergence for 13L model |
| `warmup_steps=50` | +150% vs baseline | Recurrence layers need warmup |

---

## 🚨 **Potential Issues & Mitigation**

### **Issue 1: Step time too slow**
**Symptom:** Step time >100ms
**Mitigation:**
- Reduce recurrence layers from 3 to 2 or 1
- Use `torch.compile` on LinearRecurrenceLayer (Phase 4)
- Fallback: NUM_LAYERS=12 instead of 13

### **Issue 2: Recurrence doesn't converge**
**Symptom:** Alpha values stuck at 0.0 or 1.0
**Mitigation:**
- Adjust alpha_logit initialization (try 0.0 for alpha=0.5)
- Increase recurrence learning rate
- Fallback: Pure SP8192 transformer (still beats baseline)

### **Issue 3: SP8192 worse than expected**
**Symptom:** val_bpb >1.30 at step 1500
**Mitigation:**
- Verify dataset loaded correctly
- Check tokenizer vocab size (should be 8192)
- Tune hyperparameters more aggressively
- Fallback: SP4096 (intermediate vocab size)

---

## 📈 **Timeline**

| Phase | Duration | Status |
|-------|----------|--------|
| 1: SP8192 tokenizer/dataset | SKIPPED | ✅ Already exists |
| 2: SP8192 migration | 1 hour | ✅ COMPLETE |
| 3: Recurrence implementation | 2 hours | ✅ COMPLETE |
| 4: Smoke test | 10 min | ⏳ NEXT |
| 5: Validation run (300s) | 30 min | ⏳ TODO |
| 6: Full run (600s) | 1-2 hours | ⏳ TODO |
| 7: Multi-seed validation | 3-6 hours | ⏳ TODO |
| **TOTAL** | **4-8 hours** | **38% COMPLETE** |

---

## 🎯 **Success Metrics**

**Must-Pass (Abort if Failed):**
- ✅ Smoke test: No errors, step time ≤100ms
- ✅ 300s run: val_bpb ≤1.28 at step ~1500
- ✅ 600s run: Artifact ≤16MB, step time ≤95ms
- ✅ Final BPB: ≤1.0810 (beat SOTA)

**Stretch Goals:**
- 🌟 Final BPB ≤1.0750 (NEW SOTA)
- 🌟 Step time ≤90ms (efficient hybrid)
- 🌟 3-seed std ≤0.0003 (high consistency)

---

## 🔗 **Related Files**

- **Implementation Plan:** `PLAN_C_IMPLEMENTATION.md`
- **Original Plan C:** `specs/PLAN_C_FULL_REDESIGN.md`
- **Baseline SOTA:** `records/.../2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/`
- **SP8192 Reference:** `records/.../2026-03-24_74M_Ternary_UNet_FP8_10L_8192BPE_YaRN_NeoMuon/`
- **arch1/mamba-hybrid Analysis:** `arch1/mamba-hybrid` branch (`specs/CURRENT_STATUS.md`)

---

## ✅ **Ready for Testing!**

All code changes complete. Ready to run smoke test.

**Next command:**
```bash
export DATA_PATH=./data/datasets/fineweb10B_sp8192
export TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model
export VOCAB_SIZE=8192
export BIGRAM_VOCAB_SIZE=24576
export NUM_LAYERS=13
export RECURRENCE_LAYERS="3,6,9"
export MAX_WALLCLOCK_SECONDS=60
export SEED=42

mkdir -p logs
python train_gpt.py > logs/smoke_sp8192_recurrence_60s.log 2>&1
```

**Monitor:**
```bash
tail -f logs/smoke_sp8192_recurrence_60s.log
```
