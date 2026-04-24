# Plan C Implementation — arch3/plan-c-sota

**Branch:** `arch3/plan-c-sota`
**Goal:** Achieve ≤1.0810 BPB (#1 RANKING) using SP8192 + Recurrence
**Target:** Beat current SOTA (1.0810 BPB by bigbag, April 9, 2026)

---

## 🎯 **Strategy: Incremental SP8192 Migration**

Instead of full redesign (weeks 1-6), we'll do **rapid iteration**:
- ✅ **Phase 1 SKIPPED:** SP8192 tokenizer + dataset already exist!
- **Phase 2:** Migrate baseline to SP8192 (1-2 days)
- **Phase 3:** Add simple recurrence layers (2-3 days)
- **Phase 4:** Optimize and validate (1-2 days)
- **Total:** **4-7 days** instead of 4-6 weeks

---

## 📦 **Assets Already Available**

✅ **SP8192 Tokenizer:**
- Location: `./data/tokenizers/fineweb_8192_bpe.model`
- Size: 363KB (fits easily in artifact)
- Vocab: 8192 tokens

✅ **SP8192 Dataset:**
- Location: `./data/datasets/fineweb10B_sp8192`
- Tokenized with SP8192 (no need to retokenize!)

✅ **Reference Implementation:**
- `records/.../2026-03-24_74M_Ternary_UNet_FP8_10L_8192BPE_YaRN_NeoMuon`
- Achieved: 1.1570 BPB with 10L × 768d model
- Uses SP8192 successfully

---

## Phase 2: SP8192 Migration (Days 1-2)

### **Objective:** Get baseline running with SP8192, no other changes

### Task 2.1: Update Data Loading

**File:** `train_gpt.py`

**Changes:**
```python
# Line ~40: Hyperparameters class
class Hyperparameters:
    # BEFORE:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))

    # AFTER:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp8192")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_8192_bpe.model")
    vocab_size = int(os.environ.get("VOCAB_SIZE", 8192))
```

**Test:** Verify data loads correctly
```bash
python -c "
import train_gpt
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load('./data/tokenizers/fineweb_8192_bpe.model')
print(f'Vocab size: {sp.GetPieceSize()}')  # Should be 8192
"
```

---

### Task 2.2: Update BigramHash for SP8192

**BigramHash vocab must scale with main vocab:**

```python
# Line ~98: BigramHash config
# BEFORE:
bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 3072))  # 3× vocab for 1024
bigram_dim = int(os.environ.get("BIGRAM_DIM", 112))

# AFTER:
bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 24576))  # 3× vocab for 8192
bigram_dim = int(os.environ.get("BIGRAM_DIM", 128))  # Slightly larger for SP8192
```

**Rationale:** BigramHash table size scales with vocabulary. 3× multiplier is standard.

---

### Task 2.3: Smoke Test (60s run)

**Environment:**
```bash
export DATA_PATH=./data/datasets/fineweb10B_sp8192
export TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model
export VOCAB_SIZE=8192
export BIGRAM_VOCAB_SIZE=24576
export BIGRAM_DIM=128
export MAX_WALLCLOCK_SECONDS=60
export SEED=42
```

**Run:**
```bash
python train_gpt.py > logs/smoke_sp8192_60s.log 2>&1
```

**Success Criteria:**
- ✅ No errors during data loading
- ✅ Training starts and runs for 60s
- ✅ Loss curve looks reasonable (initial loss ~9.0 for vocab=8192 vs ~7.0 for vocab=1024)
- ✅ Step time ≤ 90ms (slightly slower than baseline due to larger vocab)

---

### Task 2.4: Short Validation Run (300s)

**Run:**
```bash
export MAX_WALLCLOCK_SECONDS=300
python train_gpt.py --seed 42 > logs/sp8192_300s.log 2>&1
```

**Extract metrics:**
```bash
grep "step.*val_bpb" logs/sp8192_300s.log | tail -5
```

**Expected:**
- Step ~1500 val_bpb: **1.28-1.32** (worse than baseline due to no hyperparameter tuning yet)
- **CRITICAL:** Should be **≤1.35** (if >1.35, SP8192 migration failed)

**Why worse than baseline?**
- All hyperparameters were tuned for SP1024
- Larger vocab needs different LR, warmup, model dimensions
- This is EXPECTED — we'll fix in Phase 3

---

## Phase 3: Add Recurrence Layers (Days 2-4)

### **Objective:** Integrate 1-3 recurrence layers to close the BPB gap

### Task 3.1: Implement Simple Linear Recurrence (Option B from Plan C)

**Why NOT Mamba-SSM:**
- arch1/mamba-hybrid showed 613ms step time (7× too slow)
- Mamba CUDA kernels have dtype issues (see MEMORY.md)
- Linear recurrence is simpler, faster, proven to work

**Implementation:**

```python
class LinearRecurrenceLayer(nn.Module):
    """
    Simple exponential moving average recurrence.
    y[t] = alpha * y[t-1] + (1 - alpha) * x[t]

    Unbounded context with O(1) memory and O(n) time.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Learnable decay per dimension (init to 0.9 for reasonable timescale)
        self.alpha_logit = nn.Parameter(torch.zeros(dim))  # logit(0.9) ≈ 2.2
        self.norm = RMSNorm()

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, L, D)
        returns: (B, L, D)
        """
        residual = x
        x = self.norm(x)

        # Alpha in (0, 1) via sigmoid
        alpha = torch.sigmoid(self.alpha_logit)  # (D,)

        # Sequential scan (can be parallelized with associative scan later)
        B, L, D = x.shape
        h = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(L):
            h = alpha * h + (1 - alpha) * x[:, t]  # EMA update
            outputs.append(h)

        y = torch.stack(outputs, dim=1)  # (B, L, D)
        return residual + y
```

**Integration Points:**
- Insert after layers 3, 6, 9 (similar to Plan C suggestion)
- Use same `mamba_layers` infrastructure from arch1/mamba-hybrid
- Total: **3 recurrence layers + 10 attention layers = 13 layers**

---

### Task 3.2: Modify GPT.__init__ for Hybrid Architecture

**Add recurrence layer initialization:**

```python
# In GPT.__init__, after attention blocks:
self.recurrence_layer_set = {3, 6, 9}  # Layers with recurrence
self.recurrence_blocks = nn.ModuleList([
    LinearRecurrenceLayer(model_dim) for _ in range(len(self.recurrence_layer_set))
])
self.recurrence_idx_map = {idx: i for i, idx in enumerate(sorted(self.recurrence_layer_set))}
```

---

### Task 3.3: Modify GPT.forward for Dispatch

**Add dispatch logic:**

```python
def _forward_layer(self, layer_idx, x, ...):
    # Existing attention layer forward
    x, raw_v = attention_forward(...)

    # Add recurrence AFTER attention if this layer has it
    if layer_idx in self.recurrence_layer_set:
        rec_idx = self.recurrence_idx_map[layer_idx]
        x = self.recurrence_blocks[rec_idx](x)

    return x, raw_v
```

---

### Task 3.4: Smoke Test with Recurrence (60s)

**Environment:** Same as Task 2.3 but add:
```bash
export RECURRENCE_LAYERS="3,6,9"
```

**Success Criteria:**
- ✅ Model runs without errors
- ✅ Step time ≤ 95ms (recurrence adds ~5ms overhead)
- ✅ Loss decreases

---

### Task 3.5: Validation Run with Recurrence (300s)

**Expected:**
- Step ~1400 val_bpb: **1.25-1.29** (improvement over pure SP8192)
- Recurrence should provide **-0.02 to -0.04 BPB** improvement

---

## Phase 4: Hyperparameter Tuning (Days 4-6)

### **Objective:** Optimize for SP8192 + recurrence architecture

### Task 4.1: Critical Hyperparameter Adjustments

**Based on Plan C recommendations:**

```python
# Learning rates (larger vocab needs adjustment)
tied_embed_lr = 0.04  # Was 0.035 (higher for larger embedding table)
matrix_lr = 0.022  # Was 0.025 (slightly lower for stability with recurrence)

# Warmup/Warmdown (longer for larger vocab convergence)
warmup_steps = 50  # Was 20
warmdown_iters = 6000  # Was 4000 (longer warmdown helps larger models)

# Model dimensions (adjust for recurrence overhead)
model_dim = 512  # Keep same (or try 544 if param budget allows)
mlp_mult = 3.0  # Keep same (recurrence adds capacity without MLP increase)

# QK-Gain (CRITICAL for SOTA)
qk_gain_init = 5.25  # Was 1.5 (SOTA uses 5.25 per Plan C analysis)
```

---

### Task 4.2: Full Training Run (600s)

**Environment:**
```bash
export VOCAB_SIZE=8192
export BIGRAM_VOCAB_SIZE=24576
export BIGRAM_DIM=128
export QK_GAIN_INIT=5.25
export TIED_EMBED_LR=0.04
export MATRIX_LR=0.022
export WARMUP_STEPS=50
export WARMDOWN_ITERS=6000
export RECURRENCE_LAYERS="3,6,9"
export SEED=42
export MAX_WALLCLOCK_SECONDS=600
```

**Run:**
```bash
python train_gpt.py > logs/planc_full_600s_seed42.log 2>&1
```

---

### Task 4.3: Success Criteria (from Plan C)

**Training Metrics:**
- ✅ Step 4000 BPB ≤ 1.180 (better than baseline due to SP8192)
- ✅ Post-EMA BPB ≤ 1.110
- ✅ Post-GPTQ BPB ≤ 1.085
- ✅ Artifact size ≤ 16,000,000 bytes
- ✅ Step time ≤ 100ms

**Final Metrics (with TTT):**
- 🎯 **Target: ≤1.0810 BPB** (BEAT SOTA!)
- 🎯 **Stretch: ≤1.0750 BPB** (NEW SOTA!)

---

### Task 4.4: Add Legal TTT (if needed)

**If post-GPTQ BPB is 1.081-1.085:**

Implement score-first TTT from Plan A:

```python
def legal_ttt_eval(model, val_data):
    """Score-first, single-pass TTT (legal per competition rules)"""
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

    total_loss = 0.0
    for batch in val_data:
        x, y = batch
        # Score FIRST (before any weight update)
        with torch.no_grad():
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            total_loss += loss.item()

        # THEN adapt (only affects future tokens)
        optimizer.zero_grad()
        train_loss = F.cross_entropy(model(x), y)
        train_loss.backward()
        optimizer.step()

    return total_loss / len(val_data)
```

**Expected improvement:** -0.003 to -0.007 BPB

---

## Phase 5: Multi-Seed Validation (Days 6-7)

### **Objective:** Statistical validation for submission

### Task 5.1: Run 3 Seeds (42, 1337, 2025)

```bash
for SEED in 42 1337 2025; do
    export SEED=$SEED
    python train_gpt.py > logs/planc_final_seed${SEED}.log 2>&1
done
```

---

### Task 5.2: Extract and Analyze Results

```bash
# Extract final BPB from each log
grep "final_int6_sliding_window" logs/planc_final_seed*.log

# Compute statistics
python -c "
import numpy as np
bpbs = [1.0812, 1.0809, 1.0815]  # Example (replace with actual)
print(f'Mean: {np.mean(bpbs):.4f}')
print(f'Std: {np.std(bpbs):.4f}')
print(f'Min: {np.min(bpbs):.4f}')
print(f'Max: {np.max(bpbs):.4f}')
"
```

**Success:** Mean ≤ 1.0810 BPB with p < 0.01 improvement vs SOTA (Welch's t-test)

---

### Task 5.3: Create Submission

**If successful (≤1.0810 BPB):**

```bash
# Create submission directory
mkdir -p records/track_10min_16mb/2026-04-24_SP8192_Recurrence_SOTA

# Copy files
cp train_gpt.py records/.../2026-04-24_SP8192_Recurrence_SOTA/
cp logs/planc_final_seed*.log records/.../2026-04-24_SP8192_Recurrence_SOTA/

# Write submission.json
cat > records/.../2026-04-24_SP8192_Recurrence_SOTA/submission.json <<EOF
{
  "track": "10min_16mb",
  "name": "SP8192 + Linear Recurrence",
  "val_bpb": 1.0805,
  "val_bpb_std": 0.0003,
  "seeds": [42, 1337, 2025],
  "description": "13L hybrid: 10 attention + 3 linear recurrence, SP8192 tokenizer, QK-Gain 5.25"
}
EOF
```

---

## Quick Reference — Environment Variables

```bash
# SP8192 Configuration
export DATA_PATH=./data/datasets/fineweb10B_sp8192
export TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model
export VOCAB_SIZE=8192
export BIGRAM_VOCAB_SIZE=24576
export BIGRAM_DIM=128

# Architecture
export NUM_LAYERS=13
export RECURRENCE_LAYERS="3,6,9"
export QK_GAIN_INIT=5.25

# Training
export TIED_EMBED_LR=0.04
export MATRIX_LR=0.022
export WARMUP_STEPS=50
export WARMDOWN_ITERS=6000

# Meta
export SEED=42
export MAX_WALLCLOCK_SECONDS=600
```

---

## Risk Mitigation

### **Risk 1: SP8192 performance worse than expected**
**Mitigation:**
- Reference implementation achieved 1.1570 BPB with SP8192
- We have better architecture (13L vs 10L, better quantization)
- Fallback: Use SP4096 (intermediate vocab size)

### **Risk 2: Recurrence doesn't converge**
**Mitigation:**
- Linear recurrence is simpler than Mamba (lower risk)
- Start with 1 layer, add more if successful
- Fallback: Pure SP8192 transformer (should still beat baseline)

### **Risk 3: Step time too slow**
**Mitigation:**
- Linear recurrence adds ~3-5ms per layer (manageable)
- 13 layers × 90ms = ~6600 steps in 600s (sufficient)
- Fallback: Reduce to 12 layers if needed

---

## Timeline Summary

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| 2: SP8192 Migration | 1-2 days | Baseline running with SP8192 |
| 3: Add Recurrence | 2-3 days | 13L hybrid architecture working |
| 4: Hyperparameter Tuning | 1-2 days | Full 600s run with tuned params |
| 5: Multi-Seed Validation | 1-2 days | 3-seed submission ready |
| **Total** | **5-9 days** | **Submission @ ≤1.0810 BPB** |

---

## Next Immediate Steps

1. ✅ **Clean git branch created** (`arch3/plan-c-sota`)
2. ✅ **SOTA baseline copied** (`train_gpt.py`)
3. ✅ **SP8192 assets verified** (tokenizer + dataset exist)
4. **TODO: Task 2.1** — Update data loading for SP8192
5. **TODO: Task 2.2** — Update BigramHash config
6. **TODO: Task 2.3** — Run 60s smoke test

**Ready to implement Task 2.1!**
