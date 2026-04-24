#!/bin/bash
# ===============================================================================
# RunPod Setup Script — Plan C: SP8192 + Linear Recurrence
# Branch: arch3/plan-c-sota
# Target: ≤1.0810 BPB (#1 RANKING)
# ===============================================================================

set -e

echo "=============================================="
echo " Plan C: SP8192 + Recurrence Setup"
echo "=============================================="

# -------------------------------------------------------------------------------
# 1. Clone Repository
# -------------------------------------------------------------------------------
echo ""
echo "[1/6] Cloning repository..."

if [ -d "parameter-golf" ]; then
    echo "    Repository already exists. Pulling latest..."
    cd parameter-golf
    git fetch origin
    git checkout arch3/plan-c-sota
    git pull origin arch3/plan-c-sota
else
    git clone https://github.com/johnlennyt5/parameter-golf.git
    cd parameter-golf
    git checkout arch3/plan-c-sota
    echo "    Cloned."
fi

# -------------------------------------------------------------------------------
# 2. Miniconda Installation
# -------------------------------------------------------------------------------
echo ""
echo "[2/6] Installing Miniconda..."

if [ -d "$HOME/miniconda3" ]; then
    echo "    Already installed -- skipping."
else
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b
    rm /tmp/miniconda.sh
    ~/miniconda3/bin/conda init bash
    echo "    Installed."
fi

export PATH="$HOME/miniconda3/bin:$PATH"
source ~/miniconda3/etc/profile.d/conda.sh

echo "    Accepting conda TOS..."
~/miniconda3/bin/conda config --set always_yes yes --set changeps1 no
~/miniconda3/bin/conda update -q conda 2>/dev/null || true

# -------------------------------------------------------------------------------
# 3. Python Environment
# -------------------------------------------------------------------------------
echo ""
echo "[3/6] Creating Python 3.11 environment..."

if conda env list | grep -q "^golf "; then
    echo "    Environment 'golf' already exists -- removing and recreating."
    conda env remove -n golf -y
fi

conda create -n golf python=3.11 -y
conda activate golf
echo "    Created and activated."

# -------------------------------------------------------------------------------
# 4. Install PyTorch + Dependencies
# -------------------------------------------------------------------------------
echo ""
echo "[4/6] Installing PyTorch and dependencies..."

pip install --upgrade pip -q

# Core dependencies from requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy tqdm sentencepiece huggingface-hub datasets tiktoken

echo "    PyTorch installed."

# -------------------------------------------------------------------------------
# 5. Install FlashAttention-3
# -------------------------------------------------------------------------------
echo ""
echo "[5/6] Installing FlashAttention-3..."

if python3 -c "import flash_attn" 2>/dev/null; then
    echo "    Already installed -- skipping."
else
    # Pre-compiled wheel for fast installation
    pip install --no-cache-dir flash-attn --no-build-isolation
    echo "    Installed."
fi

# -------------------------------------------------------------------------------
# 6. Download SP8192 Dataset
# -------------------------------------------------------------------------------
echo ""
echo "[6/6] Downloading SP8192 dataset..."

# Create data directory
mkdir -p ./data/datasets
mkdir -p ./data/tokenizers

# Check if dataset already exists
if [ -f "./data/datasets/fineweb10B_sp8192/fineweb_val_000.bin" ]; then
    echo "    Dataset already exists -- skipping download."
else
    echo "    Downloading from HuggingFace (this may take 10-20 minutes)..."
    # Install huggingface-cli if not present
    pip install -q huggingface-hub[cli]

    # Download dataset and tokenizer
    huggingface-cli download sproos/parameter-golf-tokenizers \
        --include "datasets/fineweb10B_sp8192/*" \
        --local-dir ./data \
        --repo-type dataset

    echo "    Downloaded."
fi

# -------------------------------------------------------------------------------
# Verification
# -------------------------------------------------------------------------------
echo ""
echo "=============================================="
echo " Verification"
echo "=============================================="

python3 - << 'EOF'
import sys
import torch
import numpy as np
import glob
import os

print(f"Python       : {sys.version.split()[0]}")
print(f"PyTorch      : {torch.__version__}")
print(f"CUDA         : {torch.cuda.is_available()}")
print(f"GPUs         : {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / (1024**3)
        print(f"  GPU {i}      : {props.name} ({mem_gb:.1f}GB)")

try:
    import flash_attn
    print(f"FlashAttn    : {flash_attn.__version__}")
except ImportError:
    print(f"FlashAttn    : NOT FOUND (may cause errors)")

# Check dataset
train_files = sorted(glob.glob("./data/datasets/fineweb10B_sp8192/fineweb_train_*.bin"))
val_files   = sorted(glob.glob("./data/datasets/fineweb10B_sp8192/fineweb_val_*.bin"))
tokenizer_exists = os.path.exists("./data/tokenizers/fineweb_8192_bpe.model")

print(f"Train shards : {len(train_files)}")
print(f"Val shards   : {len(val_files)}")
print(f"Tokenizer    : {'EXISTS' if tokenizer_exists else 'MISSING'}")

if val_files:
    total = sum(
        int(np.fromfile(f, dtype='<i4', count=3)[2])
        for f in val_files
    )
    print(f"Val tokens   : {total:,}")

# Verify code
if os.path.exists("train_gpt.py"):
    print(f"train_gpt.py : EXISTS")
    # Quick check for SP8192 and recurrence
    with open("train_gpt.py", "r") as f:
        content = f.read()
        if "fineweb10B_sp8192" in content:
            print(f"  SP8192     : CONFIGURED")
        if "LinearRecurrenceLayer" in content:
            print(f"  Recurrence : IMPLEMENTED")
else:
    print(f"train_gpt.py : MISSING (ERROR)")

EOF

echo ""
echo "=============================================="
echo " Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Activate environment: conda activate golf"
echo "  2. Run training: bash run_planc_full.sh"
echo ""
echo "Or run individual configs:"
echo "  - Smoke test (60s):  bash run_planc_smoke.sh"
echo "  - Validation (300s): bash run_planc_300s.sh"
echo "  - Full run (600s):   bash run_planc_full.sh"
echo ""
echo "=============================================="
