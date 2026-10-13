"""The L1 HTTP layer, driven by the real benchmark client (CPU only, stub backend).

Exists because the L1 smoke test on the GPU returned HTTP 422 for every request: the route
was fine in the fake server the other tests use, but not in the real FastAPI app.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from collections.abc import Iterator

import httpx
import numpy as np
import pytest
import uvicorn

from bench import client as loadgen
from server import hf_server

VOCAB = ["a ", "b ", "c ", "d ", "e "]


def stub_backend(calls: list[dict]) -> hf_server.Backend:
    """Deterministic fake: token i is VOCAB[i % 5]; records what generate() receives."""

    def encode(messages, tmpl_kwargs):
        calls.append({"messages": messages, "tmpl_kwargs": tmpl_kwargs})
        return np.zeros(7, dtype=np.int64), 7

    def decode(ids):
        return "".join(VOCAB[i % len(VOCAB)] for i in ids)

    def generate(prompt_ids, streamer, kw):
        calls[-1]["gen_kwargs"] = kw
        streamer.put(prompt_ids)  # transformers sends the prompt first; must be skipped
        n = kw["max_new_tokens"] if kw.get("min_new_tokens") else max(1, kw["max_new_tokens"] - 1)
        for i in range(n):
            time.sleep(0.001)
            streamer.put(np.array([i]))
        streamer.end()

    return hf_server.Backend("stub-model", encode, decode, generate)


def serve(backend: hf_server.Backend) -> Iterator[str]:
    """Run make_app(backend) under uvicorn in a thread; yields the base URL."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(
        uvicorn.Config(
            hf_server.make_app(backend), host="127.0.0.1", port=port, log_level="warning"
        )
    )
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if httpx.get(f"{base}/health", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.05)
    else:
        raise RuntimeError("stub hf_server did not start")
    try:
        yield base
    finally:
        srv.should_exit = True
        thread.join(timeout=5)


@pytest.fixture
def server():
    calls: list[dict] = []
    gen = serve(stub_backend(calls))
    yield next(gen), calls
    next(gen, None)


def prompts(n: int) -> list[dict]:
    return [{"id": f"p{i}", "messages": [{"role": "user", "content": f"q{i}"}]} for i in range(n)]


def run(base: str, n: int, conc: int, **gen) -> list[loadgen.RequestResult]:
    params = loadgen.GenParams(**gen)
    results, _ = asyncio.run(loadgen.run_closed_loop(base, "stub-model", prompts(n), conc, params))
    return results


def test_benchmark_client_gets_exact_token_counts(server):
    base, calls = server
    results = run(base, 5, 2, max_tokens=6, ignore_eos=True, enable_thinking=False)
    assert [r.ok for r in results] == [True] * 5, [r.error for r in results]
    assert {r.completion_tokens for r in results} == {6}
    assert {r.prompt_tokens for r in results} == {7}
    assert all(r.ttft_s is not None and r.itl_s is not None for r in results)
    assert results[0].text == "a b c d e a "
    # What the client sends is what transformers.generate() receives.
    assert calls[-1]["gen_kwargs"] == {"max_new_tokens": 6, "do_sample": False, "min_new_tokens": 6}
    assert calls[-1]["tmpl_kwargs"] == {"enable_thinking": False}


def test_finish_reason_stop_without_ignore_eos(server):
    base, _ = server
    (r,) = run(base, 1, 1, max_tokens=6, ignore_eos=False)
    assert r.ok and r.completion_tokens == 5


def test_non_streaming_is_rejected_and_models_listed(server):
    base, _ = server
    r = httpx.post(f"{base}/v1/chat/completions", json={"messages": [], "stream": False})
    assert r.status_code == 400
    assert httpx.get(f"{base}/v1/models").json()["data"][0]["id"] == "stub-model"


def test_generate_failure_is_reported_as_request_error():
    be = stub_backend([])

    def boom(prompt_ids, streamer, kw):
        raise RuntimeError("CUDA out of memory (simulated)")

    be.generate = boom
    gen = serve(be)
    base = next(gen)
    try:
        (r,) = run(base, 1, 1, max_tokens=3)
        assert r.ok is False and r.error
    finally:
        next(gen, None)
