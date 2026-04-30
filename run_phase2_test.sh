#!/bin/bash
# Phase 2 Test: All Changes Including Novel Techniques
# Expected: artifact ~15.8-16.2 MB, val_bpb 1.044-1.052

export RUN_ID=phase2_novel_all
export DATA_PATH=./data/datasets/fineweb10B_sp8192
export TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model
export VOCAB_SIZE=8192
export CASEOPS_ENABLED=0
export COMPRESSOR=pergroup

# Phase 1: Enable gates
export SMEAR_GATE_ENABLED=1
export SPARSE_ATTN_GATE_ENABLED=1

# Phase 1: LQER hyperparameter tuning
export LQER_ENABLED=1
export LQER_TOP_K=4
export LQER_RANK=6

# Phase 1: Gate INT8 quantization
export GATED_ATTN_QUANT_GATE=1

# Phase 2.1: Token-frequency-aware embedding quantization (NOVEL)
export TOKEN_FREQ_QUANT=1

# Phase 2.2: Hierarchical LQER with adaptive rank selection (NOVEL)
export LQER_ADAPTIVE_RANK=1

# Seed for reproducibility
export SEED=42

echo "=== Phase 2 Test (All Novel Techniques) ==="
echo "Phase 1 - Gates: SMEAR=$SMEAR_GATE_ENABLED, SPARSE=$SPARSE_ATTN_GATE_ENABLED"
echo "Phase 1 - LQER tuning: TOP_K=$LQER_TOP_K, RANK=$LQER_RANK"
echo "Phase 1 - Gate INT8: GATED_ATTN_QUANT_GATE=$GATED_ATTN_QUANT_GATE"
echo "Phase 2 - NOVEL: TOKEN_FREQ_QUANT=$TOKEN_FREQ_QUANT, LQER_ADAPTIVE_RANK=$LQER_ADAPTIVE_RANK"
echo "Expected: val_bpb 1.044-1.052, artifact ~15.8-16.2 MB"
echo ""

torchrun --standalone --nproc_per_node=8 train_gpt.py
