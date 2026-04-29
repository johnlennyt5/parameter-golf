#!/bin/bash
# Priority 1: Cross-Layer Sharing + KFEC
# Target: 1.036-1.041 BPB (beat SOTA 1.0611)

# Single seed validation (run this first!)
RUN_ID=priority1_seed42 \
DATA_PATH=./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
VOCAB_SIZE=8192 \
CASEOPS_ENABLED=1 \
CROSS_LAYER_SHARING=1 \
DELTA_RANK=32 \
KFEC_ENABLED=1 \
KFEC_RANK=8 \
KFEC_TOP_K=3 \
KFEC_FACTOR_BITS=4 \
KFEC_ALS_ITERS=10 \
LQER_ENABLED=0 \
SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
