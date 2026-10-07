import asyncio
import json

import pytest

from bench.client import GenParams, RequestResult, run_closed_loop
from bench.metrics import dist_ms, summarize_cell
from server import openai_format as fmt
from tests.fake_server import FakeServer

PROMPTS = [{"id": str(i), "messages": [{"role": "user", "content": f"q{i}"}]} for i in range(6)]


def test_request_body_shape():
    body = GenParams(max_tokens=16, ignore_eos=True).body("m", [{"role": "user", "content": "x"}])
    assert body["stream"] is True
    assert body["ignore_eos"] is True
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["stream_options"]["include_usage"] is True


def test_closed_loop_against_fake_server(fake_server):
    results, duration = asyncio.run(
        run_closed_loop(fake_server.base_url, "m", PROMPTS, 2, GenParams(max_tokens=5))
    )
    assert [r.id for r in results] == [p["id"] for p in PROMPTS]
    assert all(r.ok for r in results)
    r = results[0]
    assert r.completion_tokens == 5 and r.prompt_tokens == 17
    assert r.text == "t0 t1 t2 t3 t4 "
    assert 0 < r.ttft_s <= r.e2e_s
    assert r.itl_s is not None and r.itl_s > 0
    assert duration > 0
    sent = fake_server.state.bodies[0]
    assert sent["max_tokens"] == 5 and sent["stream"] is True


def test_errors_are_recorded_not_raised():
    with FakeServer(fail_every=2) as s:
        results, _ = asyncio.run(run_closed_loop(s.base_url, "m", PROMPTS, 1, GenParams(4)))
    bad = [r for r in results if not r.ok]
    assert len(bad) == 3
    assert bad[0].error.startswith("HTTP 500")


def test_concurrency_limit_is_respected():
    # 8 requests × ~20 ms each: c=1 must take noticeably longer than c=8.
    prompts = PROMPTS + PROMPTS[:2]
    with FakeServer(delay_s=0.005) as s:
        _, serial = asyncio.run(run_closed_loop(s.base_url, "m", prompts, 1, GenParams(4)))
        _, parallel = asyncio.run(run_closed_loop(s.base_url, "m", prompts, 8, GenParams(4)))
    assert serial > parallel * 2


def test_dist_ms():
    d = dist_ms([0.1, 0.2, 0.3, 0.4])
    assert d["p50"] == pytest.approx(250.0)
    assert d["mean"] == pytest.approx(250.0)
    assert d["n"] == 4
    assert dist_ms([]) is None


def test_summarize_cell():
    ok = RequestResult(
        id="a",
        ok=True,
        t_send=0,
        ttft_s=0.1,
        e2e_s=1.0,
        itl_s=0.01,
        prompt_tokens=10,
        completion_tokens=100,
    )
    bad = RequestResult(id="b", ok=False, t_send=0, error="HTTP 500")
    s = summarize_cell([ok, ok, bad], duration_s=2.0)
    assert s["num_ok"] == 2 and s["num_errors"] == 1
    assert s["output_tok_per_s"] == pytest.approx(100.0)
    assert s["req_per_s"] == pytest.approx(1.0)
    assert s["ttft_ms"]["p50"] == pytest.approx(100.0)
    assert s["errors_sample"] == ["HTTP 500"]


def test_openai_chunk_format():
    line = fmt.chunk("id1", "m", content="hi", usage_stats=fmt.usage(3, 1))
    assert line.startswith("data: ") and line.endswith("\n\n")
    body = json.loads(line[6:])
    assert body["choices"][0]["delta"] == {"content": "hi"}
    assert body["usage"]["total_tokens"] == 4
    assert json.loads(fmt.chunk("id1", "m", include_choice=False)[6:])["choices"] == []
