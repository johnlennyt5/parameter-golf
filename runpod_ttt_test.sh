#!/bin/bash
# Quick TTT Test Script for RunPod
# Run this on your 8xH100 RunPod instance

set -e  # Exit on error

echo "=================================="
echo "Parameter Golf - TTT Test Runner"
echo "=================================="
echo ""

# Check we're in the right directory
if [ ! -f "train_gpt.py" ]; then
    echo "❌ Error: train_gpt.py not found. Are you in the parameter-golf directory?"
    exit 1
fi

# Check if dataset exists
if [ ! -d "data/datasets/fineweb10B_sp8192" ]; then
    echo "📥 Downloading SP8192 dataset (this takes ~5-10 minutes)..."
    python3 data/cached_challenge_fineweb.py --variant sp8192
    echo "✅ Dataset downloaded!"
else
    echo "✅ Dataset already exists"
fi

# Verify tokenizer exists
if [ ! -f "data/tokenizers/fineweb_8192_bpe.model" ]; then
    echo "❌ Error: Tokenizer not found. Dataset download may have failed."
    exit 1
fi

# Check GPU count
GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
echo "🔍 Detected $GPU_COUNT GPUs"

if [ "$GPU_COUNT" != "8" ]; then
    echo "⚠️  WARNING: Expected 8 GPUs, found $GPU_COUNT"
    echo "   For official leaderboard, you need 8xH100 SXM"
    read -p "   Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Set environment variables
export DATA_PATH=./data/datasets/fineweb10B_sp8192/
export TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model
export VOCAB_SIZE=8192
export RUN_ID=ttt_test_$(date +%Y%m%d_%H%M%S)

echo ""
echo "🚀 Starting TTT Test Run"
echo "   Run ID: $RUN_ID"
echo "   Dataset: SP8192"
echo "   Expected time: ~10 minutes"
echo "   Expected score: 1.08-1.09 BPB"
echo ""
echo "=================================="
echo ""

# Run training
torchrun --standalone --nproc_per_node=$GPU_COUNT train_gpt.py

echo ""
echo "=================================="
echo "✅ Training Complete!"
echo ""
echo "📊 Check your results:"
echo "   Log file: logs/$RUN_ID.txt"
echo ""
echo "🔍 Look for this line in the log:"
echo '   final_int8_lzma_ttt val_bpb:X.XXXX'
echo ""
echo "✅ Success if val_bpb < 1.10"
echo "🏆 Excellent if val_bpb < 1.09"
echo "🎯 SOTA match if val_bpb ~1.08"
echo ""
echo "=================================="
echo ""

# Extract final BPB score
if [ -f "logs/$RUN_ID.txt" ]; then
    echo "📈 Final Score:"
    grep "final_int8_lzma_ttt_exact" "logs/$RUN_ID.txt" | tail -1
    echo ""
fi
