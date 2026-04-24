#!/bin/bash
# ===============================================================================
# Plan C Validation Run (300s)
# Purpose: Measure BPB improvement with SP8192 + recurrence
# ===============================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate golf

mkdir -p logs

SEED=${SEED:-42}

echo "=============================================="
echo " Plan C: Validation Run (300s)"
echo " Measuring BPB Improvement"
echo "=============================================="

RUN_ID=planc_300s_seed${SEED} \
DATA_PATH=./data/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
VOCAB_SIZE=8192 \
BIGRAM_VOCAB_SIZE=24576 \
BIGRAM_DIM=128 \
NUM_LAYERS=13 \
RECURRENCE_LAYERS="3,6,9" \
QK_GAIN_INIT=5.25 \
TIED_EMBED_LR=0.04 \
MATRIX_LR=0.022 \
WARMUP_STEPS=50 \
WARMDOWN_ITERS=6000 \
MAX_WALLCLOCK_SECONDS=300 \
VAL_LOSS_EVERY=2000 \
TRAIN_LOG_EVERY=300 \
SEED=$SEED \
OMP_NUM_THREADS=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py \
  2>&1 | tee logs/planc_300s_seed${SEED}.log

echo ""
echo "=============================================="
echo " Validation Complete!"
echo "=============================================="
echo ""
echo "Target: Step ~1500 val_bpb ≤1.28"
echo ""
echo "Extract val_bpb:"
echo "  grep 'val_bpb' logs/planc_300s_seed${SEED}.log | tail -3"
echo ""
