#!/bin/bash
# Phase 1 Test: Gates + LQER Tuning + Gate INT8 Quantization
# Expected: artifact ~16.1-16.3 MB, val_bpb 1.055-1.058

export RUN_ID=phase1_gates_lqer_int8
export DATA_PATH=./data/datasets/fineweb10B_sp8192
export TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model
export VOCAB_SIZE=8192
export CASEOPS_ENABLED=0
export COMPRESSOR=pergroup

# Phase 1.1: Enable gates
export SMEAR_GATE_ENABLED=1
export SPARSE_ATTN_GATE_ENABLED=1

# Phase 1.2: LQER hyperparameter tuning
export LQER_ENABLED=1
export LQER_TOP_K=4  # up from 3
export LQER_RANK=6   # up from 4

# Phase 1.3: Gate INT8 quantization
export GATED_ATTN_QUANT_GATE=1

# Seed for reproducibility
export SEED=42

echo "=== Phase 1 Test ==="
echo "Gates enabled: SMEAR_GATE_ENABLED=$SMEAR_GATE_ENABLED, SPARSE_ATTN_GATE_ENABLED=$SPARSE_ATTN_GATE_ENABLED"
echo "LQER tuning: TOP_K=$LQER_TOP_K, RANK=$LQER_RANK"
echo "Gate INT8 quantization: GATED_ATTN_QUANT_GATE=$GATED_ATTN_QUANT_GATE"
echo "Expected: val_bpb 1.055-1.058, artifact ~16.1-16.3 MB"
echo ""

torchrun --standalone --nproc_per_node=8 train_gpt.py
