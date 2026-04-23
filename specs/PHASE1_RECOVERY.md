# Phase 1: Recovery Implementation Summary

**Date:** 2026-04-22
**Branch:** analysis/sota-improvement
**Goal:** Fix performance regression from 1.11867 → 1.12190 BPB (+0.00323 degradation)

---

## Changes Implemented

### 1. Added Control Flags for Training Innovations

**File:** `train_gpt.py:106-108`

Added two new environment variable flags (both **disabled by default**):

```python
gradient_surgery_enabled = bool(int(os.environ.get("GRADIENT_SURGERY_ENABLED", "0")))
llrd_enabled = bool(int(os.environ.get("LLRD_ENABLED", "0")))
```

**Rationale:**
- Previous training runs had both innovations **always enabled**
- Innovations caused regression instead of improvement (see root cause analysis)
- Flags allow A/B testing and progressive re-enabling with fixes

---

### 2. Disabled Gradient Surgery (Innovation A)

**File:** `train_gpt.py:1962`

**Change:**
```python
# BEFORE:
if base_model.skip_weights.grad is not None and base_model.skip_weights.numel() > 0:

# AFTER:
if args.gradient_surgery_enabled and base_model.skip_weights.grad is not None and base_model.skip_weights.numel() > 0:
```

**Root Cause of Regression:**
1. **Dimension mismatch bug (line 1976):** Projected skip gradient against wrong 512 elements
   - `main_grad_vec[:skip_grad.numel()]` takes first 512 floats from ~2M param encoder
   - These 512 elements are just the first Q/O projection layer, NOT full encoder gradient
   - Skip connections learned to be orthogonal to **wrong target**

2. **Applied from step 1:** Premature orthogonality constraint during noisy early training
   - Encoder gradients unstable in steps 1-1000
   - Orthogonalizing against noise creates incorrect loss landscape

3. **Inverted projection logic:** Projects skip_grad away from itself, not away from main_grad

**Impact:** Visible at step 4000 (1.2059 → 1.2141 BPB, +0.0082 regression)

**Status:** **DISABLED** (default: `GRADIENT_SURGERY_ENABLED=0`)

---

### 3. Disabled Layer-Wise Learning Rate Decay (Innovation B)

**File:** `train_gpt.py:1987`

**Change:**
```python
# BEFORE:
with torch.no_grad():

# AFTER:
if args.llrd_enabled:
    with torch.no_grad():
```

**Root Cause of Regression:**
1. **Bad interaction with Muon optimizer:**
   - LLRD scales gradients **before** Muon's Newton-Schulz orthogonalization
   - Muon normalizes gradients, but 0.6× → 1.2× scaling changes **relative magnitudes**
   - Early encoder banks (0.6× LR) lag behind, creating training imbalance

2. **Grad clipping undoes LLRD effect:**
   - Scaling applied at line 2000, clipping at line 2022
   - When gradient norm > threshold, clipping **equalizes** all magnitudes
   - LLRD effect becomes **intermittent** depending on gradient norm

3. **0.6× LR too aggressive for early layers:**
   - Combined with gradient surgery bug, early encoders **barely train**
   - By step 4000, early encoders are undertrained (only 60% of expected updates)
   - Skip connections forced orthogonal to **weak encoder gradients**

**Impact:** Accumulates over 4000 steps, contributes to step 4000 regression

**Status:** **DISABLED** (default: `LLRD_ENABLED=0`)

---

### 4. Restored SOTA Bigram Configuration

**File:** `train_gpt.py:82-83`

**Change:**
```python
# BEFORE (caused -73,728 param reduction):
bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 2048))
bigram_dim = int(os.environ.get("BIGRAM_DIM", 128))

# AFTER (SOTA configuration):
bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 3072))
bigram_dim = int(os.environ.get("BIGRAM_DIM", 112))
```

**Parameter Impact:**

| Metric | Previous | Restored | Change |
|--------|----------|----------|--------|
| Bigram vocab | 2,048 | 3,072 | +1,024 (+50%) |
| Bigram dim | 128 | 112 | -16 (-12.5%) |
| Embed params | 262,144 | 344,064 | +81,920 |
| Proj params | 65,536 | 57,344 | -8,192 |
| **Total** | **327,681** | **401,409** | **+73,728** |

**Rationale:**
- 33% vocab reduction (3072 → 2048) increased bigram hash collisions
- Lost representational capacity in low-level token interactions
- Expected impact: **~0.0015-0.002 BPB improvement**

**Status:** **RESTORED** to SOTA values

---

## Innovation C Status: NOT Responsible for Regression

**QAT with Skip Connection Preservation** (lines 982-992, 1058-1063, 1522-1527)

**Timeline evidence:**
```
step:4000/20000 val_loss:2.0499 val_bpb:1.2141  ← Regression here
late_qat:enabled step:6354 scale:0.1499         ← QAT turns on LATER
```

- QAT activates at step **6354**
- Regression visible at step **4000** (2354 steps **before** QAT)
- **Innovation C is NOT the cause of step 4000 regression**

**However, Skip QAT has its own issues:**
1. **Global scale instead of per-dimension:** Non-uniform quantization noise
2. **Code duplication:** 3 identical blocks (DRY violation)

**Status:** **No changes in Phase 1** (will address in Phase 4: Quantization Refinement)

---

## Expected Performance Recovery

| Metric | Previous (Regressed) | Expected After Phase 1 | Improvement |
|--------|---------------------|------------------------|-------------|
| Step 4000 BPB | 1.2141 | ~1.206 | -0.008 |
| Post-EMA BPB | 1.1372 | ~1.134 | -0.003 |
| Final BPB | 1.12190 | ~1.1179 | -0.004 |

**Total expected recovery:** **~0.004 BPB** (returns to baseline ~1.1187)

---

## Verification Plan

### Phase 1 Test Run

**Command:**
```bash
export GRADIENT_SURGERY_ENABLED=0
export LLRD_ENABLED=0
export BIGRAM_VOCAB_SIZE=3072
export BIGRAM_DIM=112
export SEED=42

python train_gpt.py
```

**Success Criteria:**
1. ✅ Step 4000 val_bpb **< 1.21** (regression fixed)
2. ✅ Final BPB **< 1.120** (recovered to baseline)
3. ✅ Artifact size **≤ 16MB** (still valid submission)

**3-Seed Validation:**
- Seeds: 42, 314, 999
- Compute mean ± std of `final_int6_sliding_window` BPB
- Statistical test: Welch's t-test vs previous regressed run (expect p < 0.05)

---

## Next Steps

### Phase 2: Architectural Refinement (After Phase 1 validation)

**Goal:** Reach ~1.105 BPB

1. Test 13L × 480d × GQA-2 configuration
2. Implement learned skip compression (524K params)
3. Tune RoPE (base=25000, dims=32)
4. Run 3-seed validation

**Expected:** ~0.013 BPB improvement

---

### Phase 3: Training Optimization

**Goal:** Reach ~1.095 BPB

1. Implement curriculum learning (progressive seq_len)
2. Adjust SWA (start earlier at 25%, lower LR)
3. Add cosine LR schedule with restarts

**Expected:** ~0.010 BPB improvement

---

### Phase 4: Quantization Refinement

**Goal:** Reach ~1.085 BPB

1. Implement Hessian-only mixed precision (int5/6/7)
2. Test learned quantization scales during QAT
3. Fix Skip QAT per-dimension scaling

**Expected:** ~0.010 BPB improvement

---

### Phase 5: Final Polish

**Goal:** Reach **<1.0810 BPB** (SOTA target)

1. Optimize sliding window evaluation
2. Test ensemble decoding (if rules allow)
3. Run 10-seed validation for statistical significance

**Expected:** ~0.005 BPB improvement

---

## Files Modified

- ✅ `train_gpt.py` (3 changes: flags, bigram config, innovation wrapping)
- ✅ `specs/PHASE1_RECOVERY.md` (this document)

---

## Commit Message

```
[Phase 1 Recovery] Disable regressive training innovations + restore SOTA bigram config

- Add GRADIENT_SURGERY_ENABLED=0 flag (disabled by default due to dimension mismatch bug)
- Add LLRD_ENABLED=0 flag (disabled by default due to Muon interaction bug)
- Restore bigram_vocab_size=3072, bigram_dim=112 (SOTA values, +73,728 params)

Expected recovery: ~0.004 BPB (1.1219 → 1.1179)

Root cause analysis documented in specs/REGRESSION_ANALYSIS.md
Phase 1 implementation summary in specs/PHASE1_RECOVERY.md
```

---

## Risk Assessment

**Low Risk:**
- Changes are **purely subtractive** (disabling broken features, restoring known-good config)
- No new code or untested features introduced
- Worst case: No improvement (but unlikely given strong diagnostic evidence)

**Validation Required:**
- 3-seed test run to confirm expected ~0.004 BPB recovery
- If recovery < 0.002 BPB, investigate for other contributing factors

**Contingency:**
- If no recovery observed, run ablation:
  1. Test with ONLY bigram config restored (isolate +73K param effect)
  2. Test with ONLY innovations disabled (isolate training bug effect)
  3. Compare to determine which fix dominates

---

## Conclusion

Phase 1 implements **conservative, evidence-based fixes** to recover from the 1.11867 → 1.12190 BPB regression. All changes are:

1. **Reversible** (environment variable flags)
2. **Low-risk** (disabling broken code, restoring known-good config)
3. **Well-documented** (clear rationale for each change)

**Next action:** Run 3-seed validation to confirm recovery, then proceed to Phase 2.
