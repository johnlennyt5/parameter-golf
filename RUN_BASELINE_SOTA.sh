#!/bin/bash
# Baseline SOTA 1.0611 BPB - Exact reproduction
# SP8192 + LQER + SparseGate + BOSSmearFix + 9-hyperparameter stack

RUN_ID=sota_baseline_1.0611 \
DATA_PATH=./data/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
VOCAB_SIZE=8192 \
CASEOPS_ENABLED=0 \
COMPRESSOR=pergroup \
SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
