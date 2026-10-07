"""Read the committed workload / eval files (JSONL, one request per line)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from bench.config import REPO_ROOT


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with resolve(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with resolve(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "id" not in row or "messages" not in row:
                raise ValueError(f"{path}: each row needs 'id' and 'messages'")
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def describe(path: str | Path, used: int) -> dict[str, Any]:
    """Provenance block recorded in every result file."""
    return {"file": str(path), "sha256": sha256_file(path), "num_used": used}
