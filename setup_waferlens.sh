#!/usr/bin/env bash
# Mac-bound bootstrap for WaferLens-Sherlock. Run from the unzipped repo root:
#   ./setup_waferlens.sh
set -euo pipefail

YELLOW='\033[1;33m'; GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'
say() { echo -e "${YELLOW}=>${NC} $*"; }
ok()  { echo -e "${GREEN}OK${NC} $*"; }
die() { echo -e "${RED}ERROR${NC} $*" >&2; exit 1; }

say "WaferLens-Sherlock bootstrap — Mac / Apple Silicon edition"
[[ -f pyproject.toml ]] || die "Run from the unzipped waferlens/ directory."
grep -q '"waferlens"' pyproject.toml || die "This does not look like the WaferLens repo."

if ! command -v uv >/dev/null 2>&1; then
  say "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  [[ -f "$HOME/.cargo/env" ]] && source "$HOME/.cargo/env" || true
  ok "uv installed."
else
  ok "uv present ($(uv --version))"
fi

say "Creating .venv (Python 3.12)..."
uv venv --python 3.12
say "Installing waferlens + dev deps (PyTorch download can take a few minutes)..."
uv pip install -e ".[dev]"
ok "Install complete."

# shellcheck disable=SC1091
source .venv/bin/activate

say "Verifying device (MPS expected on Apple Silicon)..."
python -c "import torch; print('  MPS:', torch.backends.mps.is_available(), '| CUDA:', torch.cuda.is_available())"

say "Running tests..."
pytest -q -m "not slow" || die "Tests failed."
ok "All tests passed."

say "Running synthetic demo (no downloads): generate data -> train CNN + ViT -> compare..."
make demo
ok "Demo complete."

cat << "EOF"

  =======================================================
  WaferLens-Sherlock is ready.

  Next:
    source .venv/bin/activate

  Use the REAL dataset (MixedWM38):
    uv pip install -e '.[kaggle]'      # for Kaggle download
    # put kaggle.json in ~/.kaggle/ (chmod 600)
    waferlens download --dataset mixedwm38
    mv data/Wafer_Map_Datasets.npz data/   # if needed
    # set data.dataset: mixedwm38 and train.epochs: 40 in configs/config.yaml
    make train-cnn && make train-vit && make compare

  Interactive dashboard:
    make ui                            # http://localhost:8501

  Single-wafer FA triage report (HTML):
    waferlens triage --sample 0 --model vit
  =======================================================
EOF
