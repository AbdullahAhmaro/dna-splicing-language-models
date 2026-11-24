#!/usr/bin/env bash
# Build an isolated Python environment for the DNA phase.
# Nothing here touches the system Python: everything lands in ./dna-venv
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d dna-venv ]; then
  echo "Creating virtual environment in $(pwd)/dna-venv"
  python3 -m venv dna-venv
fi

# shellcheck disable=SC1091
source dna-venv/bin/activate

python -m pip install --upgrade pip wheel

# CPU-only PyTorch: no NVIDIA card needed, and about 1 GB smaller than the
# default wheel. If this index is blocked on your network, drop the
# --index-url line and just run: pip install torch
pip install torch --index-url https://download.pytorch.org/whl/cpu

# transformers is pinned: several DNA models on the Hub ship their own code
# written against transformers 4.x, and it breaks on 5.x.
pip install "transformers==4.49.0" "datasets" "numpy" "scikit-learn" "pandas"

# matplotlib draws the charts (05_chart.py, 11_scan_report.py); einops is needed by
# DNABERT-2 (see the note in try_dna_models.py). Both are in the working venv but
# were missing from this script; added on 2026-11-24 so a fresh clone gets a working
# environment. Exact versions of the working venv are in requirements-frozen.txt.
pip install "matplotlib" "einops"

echo
echo "Done. From now on, start each session with:"
echo "    source dna-venv/bin/activate"
python - <<'PY'
import torch, transformers
print(f"torch {torch.__version__}, transformers {transformers.__version__}")
PY
