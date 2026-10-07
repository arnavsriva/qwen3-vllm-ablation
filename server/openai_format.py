"""OpenAI-compatible streaming chunk format, shared by the L1 HF server and tests."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

DONE = "data: [DONE]\n\n"


def new_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def chunk(
    completion_id: str,
    model: str,
    *,
    content: str | None = None,
    role: str | None = None,
    finish_reason: str | None = None,
    usage_stats: dict[str, int] | None = None,
    include_choice: bool = True,
    created: int | None = None,
) -> str:
    delta: dict[str, Any] = {}
    if role is not None:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    body: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created if created is not None else int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
        if include_choice
        else [],
    }
    if usage_stats is not None:
        body["usage"] = usage_stats
    return f"data: {json.dumps(body)}\n\n"
