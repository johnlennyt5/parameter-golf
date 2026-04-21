# Implementation Checklist: Top-5 SOTA Attack

## Pre-Flight Check

- [ ] **SP4096 tokenizer downloaded** (`data/tokenizers/fineweb_4096_bpe.model`)
- [ ] **SP6144 tokenizer downloaded** (`data/tokenizers/fineweb_6144_bpe.model`)
- [ ] **SP8192 tokenizer downloaded** (`data/tokenizers/fineweb_8192_bpe.model`)
- [ ] **SP4096 dataset downloaded** (`data/datasets/fineweb10B_sp4096/`)
- [ ] **SP6144 dataset downloaded** (`data/datasets/fineweb10B_sp6144/`)
- [ ] **SP8192 dataset downloaded** (`data/datasets/fineweb10B_sp8192/`)
- [ ] **Baseline verified** (current 1.1147 BPB on SP1024)

---

## Phase 1: Quick Wins (SP1024 Baseline Improvements)

**Goal:** Validate hyperparameter tuning on existing baseline

### Changes

- [ ] **QK-Gain sweep** - Test {2.0, 3.0, 4.0, 5.0} to find optimal
  - File: `train_gpt.py` line 59
  - Expected gain: -0.002 to -0.004 BPB

- [ ] **Weight decay increase** - 0.04 → 0.085
  - File: `train_gpt.py` lines 92-93
  - Expected gain: -0.001 to -0.003 BPB

- [ ] **EMA weight averaging** - Add decay=0.997
  - File: `train_gpt.py` after line 2273
  - Expected gain: -0.001 to -0.002 BPB

### Testing

```bash
# QK-Gain sweep (4 runs)
for qk in 2.0 3.0 4.0 5.0; do
  QK_GAIN_INIT=$qk SEED=42 MAX_WALLCLOCK_SECONDS=60 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py | tee qk_${qk}.log
done

# Analyze results
python3 -c "
import re
for qk in [2.0, 3.0, 4.0, 5.0]:
    with open(f'qk_{qk}.log') as f:
        log = f.read()
        bpb = re.findall(r'val_bpb:(\d+\.\d+)', log)[-1]
        print(f'QK_GAIN={qk}: BPB={bpb}')
"
```

**Expected Phase 1 result:** 1.1147 → 1.1060-1.1090 BPB

---

## Phase 2: Novel Architecture (SP4096 + MoD Recurrence)

**Goal:** Implement 3 novel techniques on SP4096 tokenizer

### Change 1: Curriculum Vocabulary Switching

**File:** `train_gpt.py` lines 60-62

```python
# Add vocab schedule
vocab_schedule = os.environ.get("VOCAB_SCHEDULE", "4096@0,6144@2000,8192@4000")
current_vocab_size = int(os.environ.get("VOCAB_SIZE", 8192))

def parse_vocab_schedule(schedule_str):
    stages = []
    for stage in schedule_str.split(','):
        vocab, step = stage.split('@')
        stages.append((int(step), int(vocab)))
    return sorted(stages)

vocab_stages = parse_vocab_schedule(vocab_schedule)
```

- [x] Code added (lines 79-95 in train_gpt.py)
- [x] Tested locally (syntax check passed)

### Change 2: Mixture-of-Depths Recurrence Config

**File:** `train_gpt.py` lines 985-990

```python
# Recurrence configuration (layer → loop count)
self.recurrence_config = {
    2: 2,  # Layer 2: loop 2×
    3: 3,  # Layer 3: loop 3× (novel)
    4: 3,  # Layer 4: loop 3× (novel)
    5: 2,  # Layer 5: loop 2×
}
```

- [x] Code added (lines 865-872 in train_gpt.py, GPT.__init__)
- [x] Tested locally (syntax check passed)

### Change 3: Update Forward Pass for MoD

**File:** `train_gpt.py` lines 1133-1145

```python
for i in range(self.num_encoder_layers):
    if i in self.recurrence_config:
        loop_count = self.recurrence_config[i]
        for loop_iter in range(loop_count):
            self.current_loop_iter = loop_iter
            x, raw_v = self._forward_layer(i, x, x0, input_ids, ve_cache, v0)
    else:
        x, raw_v = self._forward_layer(i, x, x0, input_ids, ve_cache, v0)
```

- [x] Code added (lines 976-987 in train_gpt.py, GPT.forward() and forward_logits())
- [x] Tested locally (syntax check passed)

### Change 4: Vocab Switching in Training Loop

**File:** `train_gpt.py` lines 2280-2310

```python
# In main training loop, check if we need to switch vocab
for step in range(iterations):
    # Check if we've hit a vocab switch point
    for switch_step, target_vocab in vocab_stages:
        if step == switch_step and current_vocab_size != target_vocab:
            log0(f"step:{step} switching vocab {current_vocab_size} → {target_vocab}")

            # Extend embedding matrix
            old_emb = base_model.tok_emb.weight.data.clone()
            new_emb = torch.nn.Embedding(target_vocab, model_dim).to(device)
            new_emb.weight.data[:current_vocab_size] = old_emb

            # Initialize new rows via interpolation
            for i in range(current_vocab_size, target_vocab):
                neighbors = torch.randint(0, current_vocab_size, (5,))
                new_emb.weight.data[i] = old_emb[neighbors].mean(dim=0)

            base_model.tok_emb = new_emb
            current_vocab_size = target_vocab

            # Reload tokenizer and dataset
            tokenizer_path = f"./data/tokenizers/fineweb_{target_vocab}_bpe.model"
            data_path = f"./data/datasets/fineweb10B_sp{target_vocab}"
            sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
            train_loader = DistributedTokenLoader(f"{data_path}/fineweb_train_*.bin", rank, world_size, device)
```

- [x] Code added (lines 1951-1971 in train_gpt.py, training loop)
- [x] Tested locally (syntax check passed)

### Change 5: Recurrence-Aware Quantization

**File:** `train_gpt.py` lines 2466-2490

```python
def recurrence_aware_quantize(state_dict, recurrence_config):
    """Assign bit widths based on recurrence depth."""
    bit_allocation = {}

    for layer_idx in range(11):
        if layer_idx in recurrence_config:
            loop_count = recurrence_config[layer_idx]
            if loop_count >= 3:
                bits = 8  # int8 for 3× recurrence
            elif loop_count == 2:
                bits = 7  # int7 for 2× recurrence
            else:
                bits = 6  # int6 default
        else:
            bits = 6  # int6 for non-recurrent layers

        # Apply to all weights in this layer
        for name in state_dict.keys():
            if f'layers.{layer_idx}.' in name or f'blocks.{layer_idx}.' in name:
                bit_allocation[name] = bits

    return bit_allocation
```

- [x] Code added (lines 1581-1625 in train_gpt.py, recurrence_aware_bit_allocation() + Option A mitigation)
- [x] Tested locally (syntax check passed)

### Change 6: Reduce to 10 Layers

**File:** `train_gpt.py` line 61

```python
num_layers = int(os.environ.get("NUM_LAYERS", 10))  # Was 11
```

- [x] Code added (line 50 in train_gpt.py)
- [x] Tested locally (syntax check passed)

---

## Phase 2 Testing

### Test 1: Short Validation Run (60s)

```bash
MAX_WALLCLOCK_SECONDS=60 \
VOCAB_SCHEDULE="4096@0,6144@20,8192@40" \
NUM_LAYERS=10 \
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py | tee test_60s.log
```

**Verify:**
- [ ] Vocab switches at steps 20, 40 (check log)
- [ ] Layers 3-4 loop 3× (check model forward)
- [ ] No crashes
- [ ] Memory < 80GB per GPU

### Test 2: Full 10-Min Run (1 seed first)

```bash
VOCAB_SCHEDULE="4096@0,6144@2000,8192@4000" \
NUM_LAYERS=10 \
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py | tee run_seed42.log
```

**Verify:**
- [ ] Training completes in ~600s
- [ ] Vocab switches at steps 2000, 4000
- [ ] Final artifact < 16 MB
- [ ] Final BPB < 1.10 (ideally < 1.09)

### Test 3: Full 3-Seed Validation

```bash
for seed in 314 42 999; do
  VOCAB_SCHEDULE="4096@0,6144@2000,8192@4000" \
  NUM_LAYERS=10 \
  SEED=$seed \
  torchrun --standalone --nproc_per_node=8 train_gpt.py | tee run_seed${seed}.log
done
```

**Verify:**
- [ ] All 3 seeds complete
- [ ] 3-seed mean BPB < 1.086 (top-5 SOTA)
- [ ] 3-seed std BPB < 0.001 (reproducible)
- [ ] All artifacts < 16 MB

---

## Phase 3: Analysis & Submission

### BPB Analysis

```python
import re
import numpy as np
from scipy import stats

# Extract BPB scores
seeds = [314, 42, 999]
bpbs = []

for seed in seeds:
    with open(f'run_seed{seed}.log') as f:
        log = f.read()
        # Find final sliding window BPB
        matches = re.findall(r'final_int6_sliding_window_exact val_bpb:(\d+\.\d+)', log)
        if matches:
            bpbs.append(float(matches[-1]))

print(f"3-seed BPBs: {bpbs}")
print(f"Mean: {np.mean(bpbs):.5f}")
print(f"Std: {np.std(bpbs, ddof=1):.5f}")

# Compare vs SOTA #1 (1.0810)
sota_mean = 1.0810
t_stat, p_value = stats.ttest_1samp(bpbs, sota_mean)
print(f"vs SOTA #1: t={t_stat:.2f}, p={p_value:.6f}")

if np.mean(bpbs) < 1.086:
    print("✓ TOP-5 ACHIEVED")
if np.mean(bpbs) < 1.084:
    print("✓✓ TOP-3 ACHIEVED")
if np.mean(bpbs) < 1.081:
    print("✓✓✓ CHALLENGING #1")
```

- [ ] Mean BPB calculated
- [ ] Statistical significance verified
- [ ] Ranking determined

### Artifact Verification

```bash
# Check all 3 artifacts
for seed in 314 42 999; do
  size=$(stat -c%s final_model_seed${seed}.int6.ptz 2>/dev/null || stat -f%z final_model_seed${seed}.int6.ptz)
  echo "Seed $seed: $size bytes ($(echo "scale=2; $size / 1024 / 1024" | bc) MB)"
  if [ $size -gt 16000000 ]; then
    echo "  ✗ EXCEEDS 16 MB LIMIT"
  else
    echo "  ✓ Within limit"
  fi
done
```

- [ ] All artifacts < 16,000,000 bytes
- [ ] Margin > 50 KB (safety buffer)

### Create Submission

```python
import json

submission = {
    "val_bpb": 1.0757,  # Replace with actual 3-seed mean
    "seeds": [314, 42, 999],
    "technique_name": "MoD Recurrence + Curriculum Vocab + Recurrence-Aware Quant",
    "techniques": [
        "4-layer mixture-of-depths recurrence (L2:2×, L3-4:3×, L5:2×)",
        "Curriculum vocabulary (SP4096→SP6144→SP8192 progressive growth)",
        "Recurrence-aware layer-wise quantization (int6/int7/int8 by loop depth)",
        "10-layer model (vs 11 in baseline)",
        "SP8192 final tokenizer (via curriculum, not static)",
    ],
    "novel_contributions": [
        "First mixture-of-depths via variable loop counts per layer",
        "First curriculum vocabulary expansion in Parameter Golf",
        "First recurrence-depth-aware quantization (not Hessian-based)",
    ],
    "artifact_size_bytes": 15900000,  # Replace with actual max
    "training_time_seconds": 600,
    "evaluation_time_seconds": 180,  # Estimate
}

with open('submission.json', 'w') as f:
    json.dump(submission, f, indent=2)

print("Submission JSON created!")
```

- [ ] Submission JSON created
- [ ] All fields filled accurately
- [ ] Novel contributions clearly stated

---

## Phase 4 (Optional): Legal Score-First TTT

**Only proceed if:**
- Phase 2 achieves < 1.08 BPB
- Want to challenge #1 (1.0810 BPB)
- Have budget for additional testing

**Implementation:** See `BATTLE_PLAN.md` Section "Differentiator 2" (currently commented out)

**Expected gain:** -0.0010 to -0.0020 BPB

---

## Risk Mitigation Checklist

- [ ] **Vocab switch causes loss spike?**
  - Solution: Add LR warmup (50 steps at 0.1× LR) after each switch

- [ ] **4-layer recurrence OOM?**
  - Solution: Enable gradient checkpointing on recurrent layers only
  - Or reduce batch size 786k → 524k tokens

- [ ] **Artifact > 16 MB?**
  - Solution: Increase int5 allocation (int6→int5 on 20% of layers)
  - Or reduce final vocab 8192 → 7680

- [ ] **BPB doesn't improve?**
  - Fallback: Disable curriculum vocab, use static SP8192
  - Still have MoD + recurrence-aware quant → expect 1.0820-1.0850 BPB

---

## Success Criteria

### Minimum (Top-10)
- [ ] 3-seed mean BPB < 1.09
- [ ] Artifact < 16 MB
- [ ] Training < 600s

### Target (Top-5)
- [ ] 3-seed mean BPB < 1.086
- [ ] Artifact < 15.9 MB (margin)
- [ ] p < 0.05 vs baseline

### Stretch (Top-3 / Challenge #1)
- [ ] 3-seed mean BPB < 1.08
- [ ] Artifact < 15.8 MB
- [ ] p < 0.01 vs baseline

---

## Timeline Estimate

| Phase | Duration | Cost (RunPod) |
|-------|----------|---------------|
| Data download | 30 min | Free |
| Code implementation | 4-6 hours | Free |
| Short validation (60s) | 10 min | ~$1 |
| Full 1-seed run | 12 min | ~$8 |
| Full 3-seed runs | 36 min | ~$24 |
| Analysis & submission | 1 hour | Free |
| **Total** | **6-8 hours** | **~$33** |

---

## Notes

- Keep detailed logs of each run
- Document any unexpected behaviors
- If anything fails, fall back to static SP8192 (proven to work)
- TTT is legal but optional - validate core first
