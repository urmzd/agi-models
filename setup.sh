#!/usr/bin/env bash
# Setup registergpt on a fresh machine (e.g. RunPod)
# Usage: bash setup.sh
set -euo pipefail

cd /workspace

# Clone repos if needed
[ -d registergpt ] || git clone https://github.com/urmzd/registergpt.git
[ -d parameter-golf ] || git clone https://github.com/urmzd/parameter-golf.git

# Install uv if needed
command -v uv &>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Sync dependencies
cd /workspace/registergpt
uv sync --extra cuda

# Download data
cd /workspace/parameter-golf
uv run python data/cached_challenge_fineweb.py --variant sp1024

# Copy training script
cp /workspace/registergpt/train.py /workspace/parameter-golf/train_registergpt.py

echo "Setup complete. Run training with:"
echo "  cd /workspace/parameter-golf && source /workspace/registergpt/.venv/bin/activate"
echo "  torchrun --standalone --nproc_per_node=\$(nvidia-smi -L | wc -l) train_registergpt.py"
