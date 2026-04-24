#!/bin/bash
# ===============================================================================
# Plan C Smoke Test (60s quick validation)
# Purpose: Verify SP8192 + recurrence integration works
# ===============================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate golf

mkdir -p logs

SEED=${SEED:-42}

echo "=============================================="
echo " Plan C: Smoke Test (60s)"
echo " Verifying SP8192 + Recurrence Integration"
echo "=============================================="

RUN_ID=planc_smoke_seed${SEED} \
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
MAX_WALLCLOCK_SECONDS=60 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=100 \
SEED=$SEED \
OMP_NUM_THREADS=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py \
  2>&1 | tee logs/planc_smoke_seed${SEED}.log

echo ""
echo "=============================================="
echo " Smoke Test Complete!"
echo "=============================================="
echo ""
echo "Success criteria:"
echo "  ✓ No errors during startup"
echo "  ✓ Step time ~90-95ms"
echo "  ✓ Initial loss ~9.0-9.5"
echo "  ✓ Loss decreasing"
echo ""
echo "Check log: logs/planc_smoke_seed${SEED}.log"
echo ""
