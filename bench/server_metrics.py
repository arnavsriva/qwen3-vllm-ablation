"""Scrape vLLM's Prometheus /metrics: counter deltas per cell and peak gauges during it.

Metric names change between vLLM versions, so counters are recorded generically (every
`vllm:*_total`) and gauges are matched by substring.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections import defaultdict

import httpx

_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfNa]+)")

# substring → key in the result JSON
PEAK_GAUGES = {
    "kv_cache_usage_perc": "kv_cache_usage_max",
    "gpu_cache_usage_perc": "kv_cache_usage_max",  # older vLLM name
    "num_requests_running": "requests_running_max",
    "num_requests_waiting": "requests_waiting_max",
}


def parse_prometheus(text: str) -> dict[str, float]:
    """Metric name → value summed over all label sets (ignores _bucket/_created series)."""
    out: dict[str, float] = defaultdict(float)
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        name, value = m.group(1), m.group(3)
        if name.endswith(("_bucket", "_created")):
            continue
        try:
            out[name] += float(value)
        except ValueError:
            continue
    return dict(out)


def counter_deltas(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {
        name: after[name] - before.get(name, 0.0)
        for name in sorted(after)
        if name.startswith("vllm:") and name.endswith("_total")
    }


def peak_gauges(snapshot: dict[str, float], peaks: dict[str, float]) -> None:
    for name, value in snapshot.items():
        for sub, key in PEAK_GAUGES.items():
            if sub in name:
                peaks[key] = max(peaks.get(key, value), value)


async def fetch(client: httpx.AsyncClient, base_url: str) -> dict[str, float] | None:
    try:
        r = await client.get(f"{base_url}/metrics", timeout=5.0)
    except httpx.HTTPError:
        return None
    return parse_prometheus(r.text) if r.status_code == 200 else None


class MetricsSampler:
    """Background poller: counter deltas over the cell + max of selected gauges."""

    def __init__(self, base_url: str, interval_s: float = 1.0) -> None:
        self.base_url = base_url
        self.interval_s = interval_s
        self.peaks: dict[str, float] = {}
        self._before: dict[str, float] | None = None
        self._task: asyncio.Task | None = None
        self._client = httpx.AsyncClient()

    async def __aenter__(self) -> MetricsSampler:
        self._before = await fetch(self._client, self.base_url)
        self._task = asyncio.create_task(self._loop())
        return self

    async def _loop(self) -> None:
        while True:
            snap = await fetch(self._client, self.base_url)
            if snap:
                peak_gauges(snap, self.peaks)
            await asyncio.sleep(self.interval_s)

    async def __aexit__(self, *exc) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        after = await fetch(self._client, self.base_url)
        await self._client.aclose()
        self.deltas = counter_deltas(self._before or {}, after) if after else {}
        self.available = self._before is not None and after is not None
