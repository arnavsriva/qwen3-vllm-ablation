"""Turn per-request timings into the per-cell summary stored in results JSON."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from bench.client import RequestResult


def dist_ms(values_s: Sequence[float]) -> dict[str, float] | None:
    """p50/p95/mean of a list of seconds, reported in milliseconds."""
    if not values_s:
        return None
    arr = np.asarray(values_s, dtype=float) * 1000.0
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "mean": float(arr.mean()),
        "n": int(arr.size),
    }


def summarize_cell(results: Sequence[RequestResult], duration_s: float) -> dict[str, Any]:
    """Aggregate one (workload, concurrency) cell.

    - ttft_ms: time from send to first content token (includes queueing).
    - itl_ms: per-request mean inter-token latency, (t_last - t_first) / (tokens - 1),
      then p50/p95 across requests. Robust to servers that put several tokens in one
      chunk (e.g. speculative decoding).
    - output_tok_per_s / req_per_s: totals over the cell's wall-clock duration.
    """
    ok = [r for r in results if r.ok]
    out_tokens = sum(r.completion_tokens for r in ok)
    in_tokens = [r.prompt_tokens for r in ok if r.prompt_tokens is not None]
    return {
        "num_requests": len(results),
        "num_ok": len(ok),
        "num_errors": len(results) - len(ok),
        "errors_sample": [r.error for r in results if not r.ok][:5],
        "duration_s": duration_s,
        "ttft_ms": dist_ms([r.ttft_s for r in ok if r.ttft_s is not None]),
        "itl_ms": dist_ms([r.itl_s for r in ok if r.itl_s is not None]),
        "e2e_ms": dist_ms([r.e2e_s for r in ok if r.e2e_s is not None]),
        "output_tokens_total": out_tokens,
        "output_tok_per_s": out_tokens / duration_s if duration_s > 0 else None,
        "req_per_s": len(ok) / duration_s if duration_s > 0 else None,
        "input_tokens_mean": float(np.mean(in_tokens)) if in_tokens else None,
        "output_tokens_mean": out_tokens / len(ok) if ok else None,
    }
