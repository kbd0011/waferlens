.PHONY: install env test lint format clean smoke data train-cnn train-vit compare triage ui demo space-build deploy-hf

PYTHON := python
UV := uv

install:
	@command -v uv >/dev/null 2>&1 || { echo "Installing uv..."; curl -LsSf https://astral.sh/uv/install.sh | sh; }
	$(UV) venv --python 3.12
	$(UV) pip install -e ".[dev]"
	@echo ""
	@echo "  Done. Activate with:  source .venv/bin/activate"
	@echo "  Optional extras:      uv pip install -e '.[tracking,kaggle]'"
	@echo ""

env:
	@$(PYTHON) -c "import waferlens, torch; print('waferlens', waferlens.__version__, '| torch', torch.__version__)"
	@$(PYTHON) -c "import torch; print('MPS available:', torch.backends.mps.is_available()); print('CUDA available:', torch.cuda.is_available())"

test:
	pytest -m "not slow" -v

test-all:
	pytest -v

lint:
	ruff check src tests

format:
	ruff format src tests

clean:
	rm -rf .pytest_cache .ruff_cache dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# === Data ===

# Generate a synthetic wafer-map sample so the pipeline runs with zero downloads
smoke-data:
	$(PYTHON) -m waferlens.cli make-synthetic --n 2000 --out data/synthetic_sample.npz

# Download the real datasets (requires Kaggle credentials / manual Drive step)
data:
	$(PYTHON) -m waferlens.cli download --dataset mixedwm38

# === Training ===

train-cnn:
	$(PYTHON) -m waferlens.cli train --model cnn

train-vit:
	$(PYTHON) -m waferlens.cli train --model vit

compare:
	$(PYTHON) -m waferlens.cli compare

triage:
	$(PYTHON) -m waferlens.cli triage --sample 0

ui:
	streamlit run src/waferlens/ui/app.py

# === End-to-end smoke demo (synthetic data, no downloads, fast) ===

demo: smoke-data
	$(PYTHON) -m waferlens.cli train --model cnn --epochs 3 --data data/synthetic_sample.npz
	$(PYTHON) -m waferlens.cli train --model vit --epochs 3 --data data/synthetic_sample.npz
	$(PYTHON) -m waferlens.cli compare
	@echo ""
	@echo "  DEMO COMPLETE (synthetic data)."
	@echo "  For real results: make data && make train-cnn && make train-vit && make compare"
	@echo "  Launch UI:        make ui"

# === Hugging Face Space ===

space-build:
	$(PYTHON) scripts/deploy_space.py --build

deploy-hf:
	$(PYTHON) scripts/deploy_space.py --deploy --repo-id kbdev0011/waferlens
