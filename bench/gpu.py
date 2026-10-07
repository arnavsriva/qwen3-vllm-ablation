"""Peak GPU memory during a cell, sampled with nvidia-smi (no-op on machines without it)."""

from __future__ import annotations

import shutil
import subprocess
import threading


def query_memory_used_mib() -> list[int] | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    return [int(x) for x in out.split() if x.strip().isdigit()]


class GpuMemorySampler:
    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval_s = interval_s
        self.peak_mib: int | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            used = query_memory_used_mib()
            if used:
                total = sum(used)
                self.peak_mib = total if self.peak_mib is None else max(self.peak_mib, total)
            self._stop.wait(self.interval_s)

    def __enter__(self) -> GpuMemorySampler:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
