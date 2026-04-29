#!/bin/bash
# Priority 1: Cross-Layer Sharing + KFEC (3-seed validation)
# Run this only after single-seed validation succeeds

for SEED in 0 42 1234; do
    echo "========================================"
    echo "Running Priority 1 with seed=$SEED"
    echo "========================================"

    RUN_ID=priority1_seed${SEED} \
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
    SEED=$SEED \
    torchrun --standalone --nproc_per_node=8 train_gpt.py

    echo ""
    echo "Seed $SEED complete!"
    echo ""
done

echo "========================================="
echo "All 3 seeds complete!"
echo "========================================="
