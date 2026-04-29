# Competition Submission - Clean TTT Version

## 🎯 Goal
Create a working submission that:
- ✅ Fits under 16MB compressed
- ✅ Uses Test-Time Training for better scores
- ✅ Stable training (no layer looping bugs)
- ✅ Competitive BPB score

## 📊 What Changed from Previous Broken Version

### ❌ **Removed (Broken):**
- Layer looping - Caused val_bpb spike from 1.16 → 2.72
- Heavy architecture (9 layers, 4x MLP)

### ✅ **Kept (Working):**
- Test-Time Training (TTT) with LoRA
- QK-Gain 5.25
- Parallel residuals (layer 7+)
- SP8192 tokenizer

### ⚙️ **New Config:**
- **Layers:** 8 (was 9)
- **MLP mult:** 3 (was 4)
- **Looping:** DISABLED (was causing catastrophic failure)
- **Expected size:** ~15MB (was 26MB)

---

## 🚀 Run Competition Submission

### On RunPod (8xH100):

```bash
# Pull latest clean version
cd /workspace/parameter-golf
git pull origin arch4/sp8192-recurrence

# Verify you have the clean version
git log --oneline -1
# Should show: "[CLEAN VERSION] Disable layer looping..."

# Run clean competition submission
RUN_ID=competition_clean_ttt \
DATA_PATH=./data/datasets/fineweb10B_sp8192/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
VOCAB_SIZE=8192 \
SEED=137 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

---

## 📊 Expected Results

### **Size Check:**
```
Total submission size int8+lzma: ~15-16MB  ← MUST be < 16,000,000 bytes
```

### **Performance:**
```
final_int8_lzma_roundtrip val_bpb: ~1.10-1.15
final_int8_lzma_ttt val_bpb: ~1.08-1.12  ← Competition score
```

---

## ✅ Success Criteria

1. ✅ **Size:** Compressed artifact < 16,000,000 bytes
2. ✅ **Stability:** No val_bpb spikes during training
3. ✅ **TTT runs:** `final_int8_lzma_ttt` line appears
4. ✅ **TTT helps:** TTT score < roundtrip score
5. ✅ **Competitive:** val_bpb < 1.20

---

## 🎲 Multiple Seeds for Leaderboard

Competition requires 3 runs with different seeds for statistical significance:

```bash
# Run 1 (seed 137)
RUN_ID=submission_seed137 \
SEED=137 \
DATA_PATH=./data/datasets/fineweb10B_sp8192/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
VOCAB_SIZE=8192 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

# Run 2 (seed 314)
RUN_ID=submission_seed314 \
SEED=314 \
DATA_PATH=./data/datasets/fineweb10B_sp8192/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
VOCAB_SIZE=8192 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

# Run 3 (seed 271)
RUN_ID=submission_seed271 \
SEED=271 \
DATA_PATH=./data/datasets/fineweb10B_sp8192/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
VOCAB_SIZE=8192 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

---

## 📈 Check Results

```bash
# Extract BPB scores from all runs
for log in logs/submission_seed*.txt; do
    echo "=== $log ==="
    grep "final_int8_lzma_ttt_exact" "$log"
done

# Calculate mean and std
python3 -c "
import numpy as np
scores = []
for seed in [137, 314, 271]:
    with open(f'logs/submission_seed{seed}.txt') as f:
        for line in f:
            if 'final_int8_lzma_ttt_exact val_bpb:' in line:
                bpb = float(line.split('val_bpb:')[1].strip())
                scores.append(bpb)
                break
print(f'BPB scores: {scores}')
print(f'Mean: {np.mean(scores):.6f}')
print(f'Std: {np.std(scores):.6f}')
"
```

---

## 🏆 Submission Checklist

- [ ] All 3 runs completed successfully
- [ ] All runs fit under 16MB
- [ ] TTT evaluation ran on all runs
- [ ] Mean BPB is competitive (< 1.20)
- [ ] Saved all log files
- [ ] Ready to create PR with results

---

## 🚨 If Issues Occur

### **Issue: Still > 16MB**
Try:
```bash
NUM_LAYERS=7 MLP_MULT=3 ... (same command)
```

### **Issue: TTT crashes**
Check log for errors after "final_int8_lzma_roundtrip"

### **Issue: TTT makes score worse**
This is a known issue - TTT might not help on smaller models
Submit roundtrip score instead (still competitive)

---

## 📞 Next Steps After Successful Run

1. ✅ Verify size < 16MB
2. ✅ Verify 3-run mean BPB
3. ✅ Copy logs to local machine
4. ✅ Prepare PR for OpenAI repo
5. 🚀 Submit to leaderboard!
