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
| GPU | 1× NVIDIA L4 (24 GB) on [Modal](https://modal.com) (`gpu="L4"`) |
| Driver / CUDA | _recorded per run in `results/<level>/*.json`_ |
| vLLM / torch / transformers | vLLM pinned in [`cloud/modal_app.py`](cloud/modal_app.py); exact versions recorded per run |

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

- **Workloads** are fixed files built once from public datasets with seed 0
  ([sources](bench/workloads/SOURCES.md)):
  - `chat`: 256 UltraChat prompts
  - `rag`: 64 SQuAD v2 questions with ~3–3.6k tokens of passages
  - `shared_prefix`: 128 questions behind one ~2k-token system prompt (used from L5 on)
- **Same output length at every level:** temperature 0, `max_tokens=256`, and the model
  is not allowed to stop early (`ignore_eos`). Qwen3 "thinking" mode is off.
- **Closed-loop concurrency sweep:** 1, 4, 16, 64 requests in flight. L1 serves one
  request at a time, so its higher-concurrency cells measure queueing.
- **Metrics per cell:**
  - TTFT p50/p95: send → first token, including queueing
  - inter-token latency p50/p95: per-request `(t_last − t_first) / (tokens − 1)`
  - output tokens/s and requests/s over the cell's wall-clock time
  - peak GPU memory (`nvidia-smi`)
  - vLLM's KV-cache usage and counter deltas (preemptions, prefix-cache hits,
    spec-decode acceptance)
- **No cache carry-over:** the prefix cache is reset before every cell, so a cell can't
  benefit from the previous one.
- **Quality on every level:**
  - GSM8K accuracy (fixed 100 questions)
  - exact greedy-output agreement with the bf16 vLLM reference (L2) on 50 prompts
- **Provenance:** every run writes `results/<level>/<timestamp>.json` with GPU, driver,
  CUDA, vLLM/torch/transformers versions, git SHA, the full resolved config, the server
  command line, matching server log lines (e.g. which attention backend was actually
  selected), and sha256 hashes of the workload files.
- **Crash-safe:** results are saved after every cell. Re-running the same command resumes.

## Results

<!-- RESULTS:START -->
_No results yet. This section will be generated from `results/` by `make plots`._
<!-- RESULTS:END -->

## Reproduce

The GPU side runs on [Modal](https://modal.com): one function call per level, with the
vLLM server and the load generator in the same container so latencies never include the
internet. Weights and results live on Modal Volumes.

```bash
cp .env.example .env        # optional: HF_TOKEN (Qwen3 is public; a token avoids rate limits)
make setup                  # dev + bench + Modal client deps (CPU)
modal setup                 # once: authenticate the Modal CLI
make modal-smoke LEVEL=L2   # tiny end-to-end check on an L4 (results/_smoke/, not committed)
make modal-bench LEVEL=L3   # one level in the background: server + every cell + quality
make modal-sweep LEVEL=L5   # a level's parameter sweep (L3 backends, L5 scheduler, L7 k)
make modal-bench-all        # all levels, sequentially
make modal-logs             # follow a background run
make pull-results           # copy finished result JSONs into results/
make plots                  # summary table + plots + README results, from results/ only
make down                   # stop any running server / Modal app
make workloads              # (optional) rebuild the workload files from source datasets
```

The same harness runs on any machine with an NVIDIA GPU: `make setup-gpu`, then
`make smoke|bench|sweep|bench-all LEVEL=…` (no Modal involved).

## Lessons

_To be written from the actual runs._

## License

MIT. See [LICENSE](LICENSE).
