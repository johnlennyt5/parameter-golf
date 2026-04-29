#!/bin/bash
# SOTA Baseline (for comparison)
# This is the current best (1.0611 BPB) - PR #1797
# Disables Priority 1 features (cross-layer sharing + KFEC)

RUN_ID=sota_baseline_seed42 \
DATA_PATH=./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
VOCAB_SIZE=8192 \
CASEOPS_ENABLED=1 \
CROSS_LAYER_SHARING=0 \
KFEC_ENABLED=0 \
SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
