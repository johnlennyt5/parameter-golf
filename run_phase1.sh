#!/bin/bash
# Phase 1 Recovery Test - Single Run

set -e

# Phase 1 recovery flags
export GRADIENT_SURGERY_ENABLED=0
export LLRD_ENABLED=0
export BIGRAM_VOCAB_SIZE=3072
export BIGRAM_DIM=112

# Data paths
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"

# Seed
export SEED="${SEED:-42}"

echo "=================================="
echo "Phase 1 Recovery Test"
echo "=================================="
echo "GRADIENT_SURGERY_ENABLED: $GRADIENT_SURGERY_ENABLED"
echo "LLRD_ENABLED: $LLRD_ENABLED"
echo "BIGRAM_VOCAB_SIZE: $BIGRAM_VOCAB_SIZE"
echo "BIGRAM_DIM: $BIGRAM_DIM"
echo "SEED: $SEED"
echo "=================================="
echo ""

python train_gpt.py
