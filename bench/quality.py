"""Quality checks so a speedup can't hide accuracy loss.

- GSM8K (fixed 100-question subset): exact-match accuracy on the final number.
- Greedy agreement: outputs on a fixed prompt set compared with a reference level (bf16
  vLLM). Quantization (L4) is expected to drift; speculative decoding (L7) should not.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

_NUM = r"-?\d[\d,]*(?:\.\d+)?"
_ANSWER_IS = re.compile(rf"answer is[:\s]*\$?\s*({_NUM})", re.IGNORECASE)
_BOXED = re.compile(rf"\\boxed\{{\s*\$?({_NUM})")
_ANY_NUM = re.compile(_NUM)


def normalize_number(s: str) -> str:
    s = s.replace(",", "").strip().rstrip(".")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def extract_answer(text: str) -> str | None:
    """Prefer 'The answer is N', then \\boxed{N}, then the last number in the text."""
    for pat in (_ANSWER_IS, _BOXED):
        found = pat.findall(text)
        if found:
            return normalize_number(found[-1])
    nums = _ANY_NUM.findall(text)
    return normalize_number(nums[-1]) if nums else None


def gsm8k_score(rows: Sequence[dict[str, Any]], outputs: dict[str, str]) -> dict[str, Any]:
    correct = 0
    missing = 0
    for row in rows:
        out = outputs.get(str(row["id"]))
        if out is None:
            missing += 1
            continue
        if extract_answer(out) == normalize_number(str(row["answer"])):
            correct += 1
    n = len(rows)
    return {"n": n, "correct": correct, "missing": missing, "accuracy": correct / n if n else None}


def common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def agreement(candidate: dict[str, str], reference: dict[str, str]) -> dict[str, Any]:
    """Exact-match rate and mean shared-prefix fraction over ids present in both."""
    ids = sorted(set(candidate) & set(reference))
    if not ids:
        return {"n": 0, "exact_match_rate": None, "mean_prefix_fraction": None}
    exact = sum(candidate[i] == reference[i] for i in ids)
    fracs = [common_prefix_len(candidate[i], reference[i]) / max(len(reference[i]), 1) for i in ids]
    return {
        "n": len(ids),
        "exact_match_rate": exact / len(ids),
        "mean_prefix_fraction": sum(fracs) / len(fracs),
    }
