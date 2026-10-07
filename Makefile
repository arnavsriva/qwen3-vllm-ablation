PY      ?= python3
VENV    ?= .venv
BIN     := $(VENV)/bin
LEVEL   ?= L2
LEVELS  := L1 L2 L3 L4 L5 L6 L7
MODAL   := $(BIN)/modal
MODAL_APP := cloud/modal_app.py
APP     := qwen3-l4-bench

.PHONY: help setup setup-gpu lint fmt test workloads run-level bench smoke sweep bench-all plots down clean \
        modal-smoke modal-bench modal-sweep modal-bench-all modal-logs pull-results modal-teardown

help:  ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n",$$1,$$2}'

setup:  ## Create venv with dev + bench + cloud (Modal client) deps, install pre-commit hooks
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -U pip
	$(BIN)/pip install -e ".[dev,bench,cloud]"
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

workloads:  ## Rebuild the committed workload/eval files from public datasets (seed 0)
	$(BIN)/pip install -e ".[workloads]"
	$(BIN)/python scripts/build_workloads.py

run-level:  ## Start the server for one level: make run-level LEVEL=L3
	$(BIN)/python -m server.launch --level $(LEVEL)

bench:  ## Benchmark one level: make bench LEVEL=L3
	$(BIN)/python -m bench.run --level $(LEVEL)

smoke:  ## Tiny run of one level → results/_smoke/ (git-ignored): make smoke LEVEL=L2
	$(BIN)/python -m bench.run --level $(LEVEL) --quick

sweep:  ## Run a level's sweep grid: make sweep LEVEL=L5
	$(BIN)/python -m bench.run --level $(LEVEL) --sweep

bench-all:  ## Benchmark every level L1..L7 sequentially
	$(BIN)/python -m bench.run --level all

plots:  ## Regenerate summary table + plots from results/
	$(BIN)/python -m bench.report --update-readme

down:  ## Stop a local server started by run-level, and stop the Modal app if one is running
	-$(BIN)/python -m server.launch --stop
	-$(MODAL) app stop $(APP) -y

# ---- Modal (1x L4; billed per second only while a function runs) ----
modal-smoke:  ## Modal: tiny end-to-end run of one level → results/_smoke/: make modal-smoke LEVEL=L2
	$(MODAL) run $(MODAL_APP) --level $(LEVEL) --quick

modal-bench:  ## Modal: one full level in the background, then `make pull-results`: make modal-bench LEVEL=L3
	$(MODAL) run --detach $(MODAL_APP) --level $(LEVEL) --background

modal-sweep:  ## Modal: a level's sweep grid in the background: make modal-sweep LEVEL=L5
	$(MODAL) run --detach $(MODAL_APP) --level $(LEVEL) --sweep --background

modal-bench-all:  ## Modal: every level L1..L7 in the background (one function call, resumes per cell)
	$(MODAL) run --detach $(MODAL_APP) --level all --background

modal-logs:  ## Stream logs of the running Modal app
	$(MODAL) app logs $(APP)

pull-results:  ## Copy finished result JSONs from the Modal results volume into results/
	$(MODAL) run $(MODAL_APP) --pull

modal-teardown:  ## Delete the Modal volumes (model cache + results copy). Local results/ is untouched.
	-$(MODAL) volume delete qwen3-bench-hf-cache -y
	-$(MODAL) volume delete qwen3-bench-vllm-cache -y
	-$(MODAL) volume delete qwen3-bench-results -y

clean:  ## Remove caches (does NOT touch results/)
	rm -rf .pytest_cache .ruff_cache **/__pycache__
