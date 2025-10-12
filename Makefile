.PHONY: help setup prep train db-sync fmt clean

# Detect uv (preferred) or fallback to system python/pip
PY_RUN := $(shell command -v uv >/dev/null 2>&1 && echo "uv run" || echo "python")
PY_PIP := $(shell command -v uv >/dev/null 2>&1 && echo "uv pip" || echo "pip")

help:
	@echo "Targets:"
	@echo "  setup      - Install Python deps (via uv if available)"
	@echo "  prep       - Prepare dataset (pass ARGS=...)"
	@echo "  train      - Train Two-Tower (pass ARGS=...)"
	@echo "  db-sync    - Sync remote DB to local (pass ARGS=-- --yes ...)"
	@echo "  fmt        - No-op placeholder (add linters/formatters if needed)"
	@echo "Examples:"
	@echo "  make setup"
	@echo "  make prep ARGS='--mode reviews --input data/dmm_reviews_videoa_2025-10-04.csv --min-stars 4 --neg-per-pos 3'"
	@echo "  make train ARGS='--embedding-dim 256 --epochs 5'"
	@echo "  make db-sync ARGS='-- --yes'"

setup:
	$(PY_PIP) install -r scripts/requirements-train.txt

prep:
	$(PY_RUN) scripts/prep_two_tower_dataset.py $(ARGS)

train:
	$(PY_RUN) scripts/train_two_tower.py $(ARGS)

db-sync:
	cd frontend && npm run db:sync:remote $(ARGS)

fmt:
	@echo "Add formatters here (e.g., ruff/black/eslint)"

clean:
	rm -rf artifacts data/interactions_*.parquet

