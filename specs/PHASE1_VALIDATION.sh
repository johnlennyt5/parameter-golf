#!/bin/bash
# Phase 1 Recovery Validation Script
#
# This script runs 3-seed validation to confirm the regression fix
# Expected improvement: ~0.004 BPB (1.1219 → 1.1179)

set -e

echo "=========================================="
echo "Phase 1 Recovery - 3-Seed Validation"
echo "=========================================="
echo ""

# Export Phase 1 recovery flags
export GRADIENT_SURGERY_ENABLED=0
export LLRD_ENABLED=0
export BIGRAM_VOCAB_SIZE=3072
export BIGRAM_DIM=112

# Ensure data and tokenizer paths are set
if [ -z "$DATA_PATH" ]; then
    export DATA_PATH="./data/datasets/fineweb10B_sp1024"
fi

if [ -z "$TOKENIZER_PATH" ]; then
    export TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"
fi

if [ -z "$VOCAB_SIZE" ]; then
    export VOCAB_SIZE=1024
fi

echo "Configuration:"
echo "  GRADIENT_SURGERY_ENABLED: $GRADIENT_SURGERY_ENABLED"
echo "  LLRD_ENABLED: $LLRD_ENABLED"
echo "  BIGRAM_VOCAB_SIZE: $BIGRAM_VOCAB_SIZE"
echo "  BIGRAM_DIM: $BIGRAM_DIM"
echo "  DATA_PATH: $DATA_PATH"
echo "  TOKENIZER_PATH: $TOKENIZER_PATH"
echo ""

# Create output directory
mkdir -p validation_results/phase1_recovery
RESULTS_DIR="validation_results/phase1_recovery/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Run 3 seeds in sequence
SEEDS=(42 314 999)

for SEED in "${SEEDS[@]}"; do
    echo "=========================================="
    echo "Running seed: $SEED"
    echo "=========================================="

    export SEED=$SEED
    export RUN_ID="phase1_recovery_seed${SEED}"

    LOG_FILE="$RESULTS_DIR/train_seed${SEED}.log"

    echo "Logging to: $LOG_FILE"
    echo ""

    # Run training
    if python train_gpt.py 2>&1 | tee "$LOG_FILE"; then
        echo ""
        echo "✅ Seed $SEED completed successfully"
        echo ""

        # Extract key metrics
        echo "Extracting metrics for seed $SEED..."

        STEP4000_BPB=$(grep "step:4000" "$LOG_FILE" | grep -oP "val_bpb:\K[0-9.]+" || echo "N/A")
        POST_EMA_BPB=$(grep "post_ema_val" "$LOG_FILE" | tail -1 | grep -oP "val_bpb:\K[0-9.]+" || echo "N/A")
        FINAL_BPB=$(grep "final_int6_sliding_window" "$LOG_FILE" | tail -1 | grep -oP "val_bpb:\K[0-9.]+" || echo "N/A")
        ARTIFACT_MB=$(grep "Saved model" "$LOG_FILE" | grep -oP "\(\K[0-9.]+(?=MB)" || echo "N/A")

        echo "  Step 4000 BPB: $STEP4000_BPB"
        echo "  Post-EMA BPB: $POST_EMA_BPB"
        echo "  Final BPB: $FINAL_BPB"
        echo "  Artifact size: ${ARTIFACT_MB}MB"
        echo ""

        # Append to results summary
        echo "$SEED,$STEP4000_BPB,$POST_EMA_BPB,$FINAL_BPB,$ARTIFACT_MB" >> "$RESULTS_DIR/summary.csv"
    else
        echo ""
        echo "❌ Seed $SEED failed!"
        echo ""
        exit 1
    fi
done

echo "=========================================="
echo "Phase 1 Validation Complete"
echo "=========================================="
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""

# Generate summary statistics
echo "Computing summary statistics..."
echo ""

if command -v python3 &> /dev/null; then
    python3 <<EOF
import csv
import statistics

with open("$RESULTS_DIR/summary.csv") as f:
    rows = list(csv.reader(f))

seeds = [row[0] for row in rows]
step4000 = [float(row[1]) for row in rows if row[1] != "N/A"]
post_ema = [float(row[2]) for row in rows if row[2] != "N/A"]
final = [float(row[3]) for row in rows if row[3] != "N/A"]
artifacts = [float(row[4]) for row in rows if row[4] != "N/A"]

print("Summary Statistics (3 seeds):")
print("=" * 50)
print(f"Seeds: {', '.join(seeds)}")
print("")
print(f"Step 4000 BPB:")
print(f"  Mean: {statistics.mean(step4000):.5f}")
print(f"  Stdev: {statistics.stdev(step4000):.5f}")
print(f"  Min: {min(step4000):.5f}, Max: {max(step4000):.5f}")
print("")
print(f"Post-EMA BPB:")
print(f"  Mean: {statistics.mean(post_ema):.5f}")
print(f"  Stdev: {statistics.stdev(post_ema):.5f}")
print(f"  Min: {min(post_ema):.5f}, Max: {max(post_ema):.5f}")
print("")
print(f"Final BPB (int6 sliding window):")
print(f"  Mean: {statistics.mean(final):.5f}")
print(f"  Stdev: {statistics.stdev(final):.5f}")
print(f"  Min: {min(final):.5f}, Max: {max(final):.5f}")
print("")
print(f"Artifact Size (MB):")
print(f"  Mean: {statistics.mean(artifacts):.2f}")
print(f"  Max: {max(artifacts):.2f}")
print("")
print("Success Criteria:")
print(f"  ✓ Step 4000 < 1.21: {statistics.mean(step4000) < 1.21}")
print(f"  ✓ Final BPB < 1.12: {statistics.mean(final) < 1.12}")
print(f"  ✓ Artifact ≤ 16MB: {max(artifacts) <= 16.0}")
print("")

# Compare to previous regressed run
prev_final = 1.12190
improvement = prev_final - statistics.mean(final)
print(f"Improvement vs Previous (1.12190 BPB):")
print(f"  Delta: {improvement:+.5f} BPB ({improvement/prev_final*100:+.2f}%)")
print(f"  Expected: ~0.004 BPB")
print(f"  Recovery successful: {improvement >= 0.003}")
print("")
EOF
else
    echo "Python3 not available for statistics computation"
    echo "Raw results in: $RESULTS_DIR/summary.csv"
fi

echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "If recovery successful (mean BPB ~1.118):"
echo "  → Proceed to Phase 2: Architectural Refinement"
echo "  → Target: ~1.105 BPB"
echo ""
echo "If recovery unsuccessful (mean BPB > 1.120):"
echo "  → Run ablation to isolate which fix dominates"
echo "  → Check for other contributing factors"
echo ""
