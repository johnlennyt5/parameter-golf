# Quick TTT Test on RunPod - Step by Step

## Prerequisites
- RunPod account with credits ($1000 OpenAI compute grant available!)
- SSH key configured in RunPod settings

## Step 1: Launch 8xH100 Pod

1. Go to https://console.runpod.io/deploy
2. Use official Parameter Golf template: https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th
3. Select **8xH100 SXM** (for official leaderboard submission)
4. Enable SSH terminal access
5. Deploy pod (~$15-20 for 10 minutes)

## Step 2: SSH Into Pod

```bash
# RunPod will give you SSH command, something like:
ssh root@X.X.X.X -p XXXXX -i ~/.ssh/id_rsa
```

## Step 3: Clone Your Repo & Download Dataset

```bash
cd /workspace
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf

# Download SP8192 dataset (takes ~5-10 minutes)
python3 data/cached_challenge_fineweb.py --variant sp8192

# Verify dataset
ls -lh data/datasets/fineweb10B_sp8192/
ls -lh data/tokenizers/
```

## Step 4: Copy Your Updated train_gpt.py

Since you've modified train_gpt.py locally with TTT, you need to transfer it:

**Method A: Git (if you committed changes)**
```bash
git checkout arch4/sp8192-recurrence  # Your branch
git pull origin arch4/sp8192-recurrence
```

**Method B: SCP (if local changes not committed)**
```bash
# On your local WSL terminal:
scp -P XXXXX train_gpt.py root@X.X.X.X:/workspace/parameter-golf/
```

**Method C: Copy-paste (quick and dirty)**
```bash
# On RunPod, open nano
nano train_gpt.py
# Copy your entire local train_gpt.py and paste
# Ctrl+X, Y, Enter to save
```

## Step 5: Run TTT Test (10 minutes!)

```bash
# Set environment variables for SP8192
export DATA_PATH=./data/datasets/fineweb10B_sp8192/
export TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model
export VOCAB_SIZE=8192
export RUN_ID=ttt_test_$(date +%Y%m%d_%H%M%S)

# Launch training on 8xH100
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Step 6: Monitor Progress

Watch for these key log lines:
```
step:1000/20000 val_loss:X.XXXX val_bpb:X.XXXX
...
final_int8_lzma_roundtrip val_loss:X.XXXX val_bpb:X.XXXX
final_int8_lzma_ttt val_loss:X.XXXX val_bpb:X.XXXX  ← THIS IS YOUR SCORE!
```

## Step 7: Check Results

**Success Criteria:**
- `final_int8_lzma_ttt val_bpb` should be **~1.08-1.09**
- If it's **< 1.10**, TTT is working! ✅
- If it's **> 1.15**, something's wrong ❌

## Step 8: Save Logs

```bash
# Logs are in logs/<RUN_ID>.txt
ls -lh logs/

# Copy log back to local machine
# On local WSL:
scp -P XXXXX root@X.X.X.X:/workspace/parameter-golf/logs/*.txt ./
```

## Expected Output

```
step:20000/20000 val_loss:1.4XXX val_bpb:1.08XX
final_int8_lzma_roundtrip val_loss:1.4XXX val_bpb:1.08XX
final_int8_lzma_ttt val_loss:1.4XXX val_bpb:1.08XX  ← TARGET: ~1.08-1.09
```

## Troubleshooting

**If val_bpb > 1.15:**
- Check dataset path is correct (SP8192, not SP1024)
- Verify QK_GAIN_INIT=5.25 in hyperparameters
- Check TTT evaluation ran (look for "final_int8_lzma_ttt" line)

**If "out of memory" error:**
- This shouldn't happen with 8xH100 (80GB each)
- Try reducing TTT_BATCH_SIZE: `export TTT_BATCH_SIZE=32`

**If training takes > 15 minutes:**
- Check you're on 8xH100 SXM (not PCIe or fewer GPUs)
- MAX_WALLCLOCK_SECONDS should cap at 600s

## Cost Estimate

- 8xH100 SXM on RunPod: ~$18-22/hour
- 10-minute test run: ~$3-4
- Dataset download: Free (one-time, ~10 min)
- **Total first run: ~$3-4**
- **Subsequent runs: ~$3-4 each** (dataset cached)

## Next Steps After Success

If val_bpb ~1.08-1.09:
1. ✅ TTT works! Celebrate! 🎉
2. ✅ You have a competitive baseline
3. ✅ Ready to add Mamba hybrid
4. ✅ Come back and I'll implement Mamba next

If val_bpb > 1.10:
1. ❌ Debug needed
2. Share the log file with me
3. I'll help identify the issue
