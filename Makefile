PY      ?= python3
VENV    ?= .venv
BIN     := $(VENV)/bin
LEVEL   ?= L2
LEVELS  := L1 L2 L3 L4 L5 L6 L7

.PHONY: help setup setup-gpu lint fmt test run-level bench bench-all plots down clean

help:  ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n",$$1,$$2}'

setup:  ## Create venv with dev + bench deps, install pre-commit hooks
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -U pip
	$(BIN)/pip install -e ".[dev,bench]"
	$(BIN)/pre-commit install

setup-gpu:  ## (GPU host only) install vLLM / transformers stack
	$(BIN)/pip install -e ".[dev,bench,gpu]"

lint:  ## Ruff lint + format check
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

fmt:  ## Auto-fix lint + format
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

test:  ## CPU-only unit tests
	$(BIN)/pytest -m "not gpu"

run-level:  ## Start the server for one level: make run-level LEVEL=L3
	$(BIN)/python -m server.launch --level $(LEVEL)

bench:  ## Benchmark one level: make bench LEVEL=L3
	$(BIN)/python -m bench.run --level $(LEVEL)

bench-all:  ## Benchmark every level L1..L7 sequentially
	$(BIN)/python -m bench.run --level all

plots:  ## Regenerate summary table + plots from results/
	$(BIN)/python -m bench.report

down:  ## Stop any local server started by run-level (cloud teardown added once provider is chosen)
	-$(BIN)/python -m server.launch --stop

clean:  ## Remove caches (does NOT touch results/)
	rm -rf .pytest_cache .ruff_cache **/__pycache__
