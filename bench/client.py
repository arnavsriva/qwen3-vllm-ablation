"""Async streaming client + closed-loop load generator for OpenAI-compatible chat servers.

Both the L1 HF server and vLLM (L2–L7) expose /v1/chat/completions with SSE streaming,
so every level is driven by exactly this code.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx


@dataclass
class GenParams:
    max_tokens: int
    temperature: float = 0.0
    ignore_eos: bool = False
    enable_thinking: bool = False
    seed: int | None = 0

    def body(self, model: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
            "ignore_eos": self.ignore_eos,
        }
        if self.seed is not None:
            body["seed"] = self.seed
        return body


@dataclass
class RequestResult:
    id: str
    ok: bool
    t_send: float
    error: str | None = None
    ttft_s: float | None = None
    e2e_s: float | None = None
    itl_s: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int = 0
    text: str = ""
    num_chunks: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def record(self, include_text: bool = False) -> dict[str, Any]:
        d = asdict(self)
        if not include_text:
            d.pop("text")
        return d


async def stream_chat(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    request_id: str,
    messages: list[dict[str, str]],
    params: GenParams,
) -> RequestResult:
    t_send = time.perf_counter()
    res = RequestResult(id=request_id, ok=False, t_send=t_send)
    t_first: float | None = None
    t_last: float | None = None
    pieces: list[str] = []
    usage: dict[str, Any] | None = None
    try:
        async with client.stream(
            "POST", f"{base_url}/v1/chat/completions", json=params.body(model, messages)
        ) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode(errors="replace")[:300]
                res.error = f"HTTP {resp.status_code}: {body}"
                return res
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    content = (choice.get("delta") or {}).get("content")
                    if content:
                        now = time.perf_counter()
                        if t_first is None:
                            t_first = now
                        t_last = now
                        pieces.append(content)
                        res.num_chunks += 1
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        res.error = f"{type(e).__name__}: {e}"
        return res

    t_end = time.perf_counter()
    res.text = "".join(pieces)
    res.completion_tokens = int(usage["completion_tokens"]) if usage else res.num_chunks
    res.prompt_tokens = int(usage["prompt_tokens"]) if usage else None
    if t_first is None:
        res.error = "no content tokens received"
        return res
    res.ok = True
    res.ttft_s = t_first - t_send
    res.e2e_s = t_end - t_send
    if res.completion_tokens > 1 and t_last is not None:
        res.itl_s = (t_last - t_first) / (res.completion_tokens - 1)
    return res


async def run_closed_loop(
    base_url: str,
    model: str,
    prompts: Sequence[dict[str, Any]],
    concurrency: int,
    params: GenParams,
    client: httpx.AsyncClient | None = None,
) -> tuple[list[RequestResult], float]:
    """Send every prompt, keeping at most `concurrency` requests in flight.

    Returns (results in prompt order, wall-clock seconds from first send to last finish).
    """
    sem = asyncio.Semaphore(concurrency)
    own_client = client is None
    if client is None:
        limits = httpx.Limits(
            max_connections=concurrency + 4, max_keepalive_connections=concurrency
        )
        client = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=30.0), limits=limits)

    async def one(p: dict[str, Any]) -> RequestResult:
        async with sem:
            return await stream_chat(client, base_url, model, str(p["id"]), p["messages"], params)

    try:
        t0 = time.perf_counter()
        results = await asyncio.gather(*(one(p) for p in prompts))
        duration = time.perf_counter() - t0
    finally:
        if own_client:
            await client.aclose()
    return list(results), duration
