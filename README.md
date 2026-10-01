# Qwen3-8B: From Raw Weights to an Optimized vLLM Server

A reproducible, layer-by-layer ablation of LLM inference optimizations on a single
NVIDIA L4. Each level adds one optimization on top of the previous one, and every level
runs the same workload, so each change can be measured on its own, for both speed and quality.

> **Status:** scaffold only. No benchmark runs yet. Every number in this README will
> come from a saved file in [`results/`](results/). Nothing is estimated.

## Overview

- **Model:** `Qwen/Qwen3-8B` (AWQ variant from L4 onward; `Qwen/Qwen3-0.6B` as the L7 draft model)
- **Serving:** Hugging Face `transformers` baseline (L1) → vLLM OpenAI-compatible server (L2–L7)
- **Harness:** async load generator with fixed seeds, a concurrency sweep, two workloads,
  and a quality check on every run

## Hardware & Versions

| Item | Value |
|---|---|
| GPU | 1× NVIDIA L4 (24 GB), provider TBD |
| Driver / CUDA | _recorded per run in `results/<level>/*.json`_ |
| vLLM / torch / transformers | _pinned once the plan is approved_ |

## Optimization Levels

| Level | Adds | One-line explanation |
|---|---|---|
| L1 | HF `transformers` baseline | `generate()` one request at a time. No batching; the reference point. |
| L2 | vLLM defaults | PagedAttention stores the KV cache in pages so many sequences share memory. Continuous batching adds and removes requests every step. |
| L3 | FlashAttention backend | Pins the attention kernel to FlashAttention and checks in the logs that it's actually in use. |
| L4 | AWQ 4-bit | Activation-aware 4-bit weights cut memory traffic and leave more room for KV cache. |
| L5 | Scheduler tuning | Tuned `max_num_seqs`, `max_num_batched_tokens`, and chunked prefill to balance TTFT and throughput. |
| L6 | Prefix caching | Reuses KV blocks for a shared system prompt so prefill only covers new tokens. |
| L7 | Speculative decoding | Qwen3-0.6B drafts tokens and the 8B model verifies them in one pass. |

Exact settings for each level are in [`configs/`](configs/).

## Benchmark Method

- **Workloads:** short chat; long-context RAG-style; shared-system-prompt variant for L6.
- **Concurrency sweep:** 1, 4, 16, 64.
- **Determinism:** fixed seeds for prompt sampling and decoding.
- **Metrics:** TTFT p50/p95, inter-token latency, output tokens/s, requests/s, peak GPU memory.
- **Quality:** a small fixed eval set on every level, so a speedup can't hide accuracy loss (this matters most for L4 and L7).
- **Provenance:** each run writes `results/<level>/<timestamp>.json` with GPU, driver, CUDA,
  vLLM version, and the full resolved config.

## Results

_No results yet. This section will be generated from `results/` by `make plots`._

## Reproduce

```bash
cp .env.example .env        # fill in HF_TOKEN
make setup                  # dev + bench deps (CPU)
make setup-gpu              # on the GPU host
make run-level LEVEL=L3     # start one level's server
make bench LEVEL=L3         # benchmark it
make bench-all              # all levels, sequentially
make plots                  # summary table + plots
make down                   # stop servers / tear down
```

## Lessons

_To be written from the actual runs._

## License

MIT. See [LICENSE](LICENSE).
