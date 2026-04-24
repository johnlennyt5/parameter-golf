#!/bin/bash
# ===============================================================================
# Plan C Full Training Run (600s on 8×H100)
# Branch: arch3/plan-c-sota
# Target: ≤1.0810 BPB (#1 RANKING)
# ===============================================================================

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate golf

# Create logs directory
mkdir -p logs

# Set random seed (default 42, can override with: SEED=1337 bash run_planc_full.sh)
SEED=${SEED:-42}

echo "=============================================="
echo " Plan C: SP8192 + Recurrence Training"
echo " Seed: $SEED"
echo " Target: ≤1.0810 BPB (#1 RANKING)"
echo "=============================================="

# ===============================================================================
# Environment Variables — Plan C Configuration
# ===============================================================================

RUN_ID=planc_full_seed${SEED} \
DATA_PATH=./data/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
VOCAB_SIZE=8192 \
BIGRAM_VOCAB_SIZE=24576 \
BIGRAM_DIM=128 \
NUM_LAYERS=13 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=3.0 \
QK_GAIN_INIT=5.25 \
TIED_EMBED_LR=0.04 \
MATRIX_LR=0.022 \
SCALAR_LR=0.025 \
WARMUP_STEPS=50 \
WARMDOWN_ITERS=6000 \
RECURRENCE_LAYERS="3,6,9" \
ROPE_BASE=10000.0 \
LOGIT_SOFTCAP=30.0 \
TRAIN_BATCH_TOKENS=786432 \
TRAIN_SEQ_LEN=2048 \
EVAL_SEQ_LEN=2048 \
EVAL_STRIDE=64 \
TIE_EMBEDDINGS=1 \
XSA_LAST_N=13 \
ROPE_DIMS=16 \
LN_SCALE=1 \
VE_ENABLED=1 \
VE_DIM=128 \
VE_LAYERS="6,7" \
SWA_ENABLED=1 \
SWA_EVERY=50 \
LATE_QAT_THRESHOLD=0.15 \
MUON_MOMENTUM=0.99 \
MUON_BACKEND_STEPS=5 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
MUON_WD=0.04 \
ADAM_WD=0.04 \
GRAD_CLIP_NORM=0.3 \
ITERATIONS=20000 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=4000 \
TRAIN_LOG_EVERY=500 \
GPTQ_CALIB_BATCHES=256 \
GPTQ_BLOCK_SIZE=128 \
SEED=$SEED \
OMP_NUM_THREADS=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py \
  2>&1 | tee logs/planc_full_seed${SEED}.log

echo ""
echo "=============================================="
echo " Training Complete!"
echo "=============================================="
echo ""
echo "Log file: logs/planc_full_seed${SEED}.log"
echo ""
echo "Extract metrics:"
echo "  grep 'step_avg' logs/planc_full_seed${SEED}.log | tail -5"
echo "  grep 'val_bpb' logs/planc_full_seed${SEED}.log | tail -5"
echo "  grep 'final_int6_sliding_window' logs/planc_full_seed${SEED}.log"
echo ""
echo "=============================================="
