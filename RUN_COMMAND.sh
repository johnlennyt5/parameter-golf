#!/bin/bash
# RunPod Execution Script - All 5 Innovations
# Expected: ~1.0814 BPB (beat SOTA with novel architecture)

set -e  # Exit on error

echo "========================================"
echo "Novel Architectural Innovations - RunPod"
echo "========================================"
echo "Target: < 1.0820 BPB (competitive with SOTA #2)"
echo "Stretch: < 1.0810 BPB (beat SOTA #1)"
echo ""

# Environment check
echo "Step 1: Checking environment..."
if [ ! -f "data/tokenizers/fineweb_8192_bpe.model" ]; then
    echo "ERROR: SP8192 tokenizer not found!"
    echo "Run: python3 data/cached_challenge_fineweb.py --variant sp8192"
    exit 1
fi

if [ ! -d "data/datasets/fineweb10B_sp8192" ]; then
    echo "ERROR: SP8192 dataset not found!"
    echo "Run: python3 data/cached_challenge_fineweb.py --variant sp8192"
    exit 1
fi

echo "✓ SP8192 tokenizer found"
echo "✓ SP8192 dataset found"
echo ""

# Single run with seed 42
echo "Step 2: Training with all 5 innovations (seed 42)..."
echo "Expected time: ~10 minutes"
echo ""

RUN_ID=all_innovations_s42 \
  DATA_PATH=./data/datasets/fineweb10B_sp8192/ \
  TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
  VOCAB_SIZE=8192 \
  SEED=42 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee logs/all_innovations_seed42.log

echo ""
echo "Step 3: Extracting results..."
echo ""

# Extract final BPB
FINAL_BPB=$(grep "final_int6_sliding_window_exact val_bpb:" logs/all_innovations_seed42.log | tail -1 | grep -oP 'val_bpb:\K[0-9.]+')

echo "========================================"
echo "RESULTS (Seed 42)"
echo "========================================"
echo "Final BPB: $FINAL_BPB"
echo ""

# Print innovation summary from log
echo "Innovation Summary:"
grep -A 20 "INNOVATION SUMMARY" logs/all_innovations_seed42.log || echo "(Innovation summary not found in log)"
echo ""

# Decision tree
if (( $(echo "$FINAL_BPB < 1.082" | bc -l) )); then
    echo "✅ SUCCESS: BPB < 1.082 - Proceed with 3-seed validation!"
    echo ""
    echo "Run these commands for seeds 314 and 999:"
    echo ""
    echo "# Seed 314"
    echo "RUN_ID=all_innovations_s314 DATA_PATH=./data/datasets/fineweb10B_sp8192/ TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model VOCAB_SIZE=8192 SEED=314 torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee logs/all_innovations_seed314.log"
    echo ""
    echo "# Seed 999"
    echo "RUN_ID=all_innovations_s999 DATA_PATH=./data/datasets/fineweb10B_sp8192/ TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model VOCAB_SIZE=8192 SEED=999 torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee logs/all_innovations_seed999.log"
elif (( $(echo "$FINAL_BPB < 1.090" | bc -l) )); then
    echo "⚠️  COMPETITIVE: BPB in range [1.082, 1.090]"
    echo "Competitive but not top-3. Analyze logs to see which innovation underperformed."
else
    echo "❌ NEEDS DEBUG: BPB > 1.090"
    echo "Check logs for errors or failed innovations."
fi

echo ""
echo "Done!"
