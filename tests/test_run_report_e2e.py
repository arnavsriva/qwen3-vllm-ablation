"""bench.run → results JSON → bench.report, end to end against the fake server (CPU only)."""

import json

import pytest

from bench import config, report, run


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Tiny workload files + results dir in tmp; real level configs."""
    wl = tmp_path / "wl"
    wl.mkdir()
    prompts = [{"id": f"p{i}", "messages": [{"role": "user", "content": "x"}]} for i in range(6)]
    for name in ("chat", "rag", "shared_prefix", "warmup", "agree"):
        write_jsonl(wl / f"{name}.jsonl", prompts)
    write_jsonl(
        wl / "gsm.jsonl",
        [{"id": "g1", "messages": [{"role": "user", "content": "x"}], "answer": "4"}],
    )
    bench = config.load_bench()
    for name in bench["workloads"]:
        bench["workloads"][name] = {"file": str(wl / f"{name}.jsonl"), "num_requests": 4}
    bench["warmup_file"] = str(wl / "warmup.jsonl")
    bench["generation"]["max_tokens"] = 4
    bench["quality"]["gsm8k"].update(file=str(wl / "gsm.jsonl"), max_tokens=5)
    bench["quality"]["agreement"].update(file=str(wl / "agree.jsonl"), max_tokens=3)
    monkeypatch.setattr(run, "load_bench", lambda: bench)
    results = tmp_path / "results"
    monkeypatch.setattr(run, "RESULTS_DIR", results)
    return results


def test_run_writes_complete_result_with_provenance(sandbox, fake_server):
    run.main(["--level", "L2", "--base-url", fake_server.base_url, "--concurrency", "1", "2"])
    files = list((sandbox / "L2").glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    assert data["status"] == "complete"
    assert data["level"] == "L2" and data["config"]["name"] == "vllm-defaults"
    assert {(c["workload"], c["concurrency"]) for c in data["cells"]} == {
        ("chat", 1),
        ("chat", 2),
        ("rag", 1),
        ("rag", 2),
    }
    cell = data["cells"][0]
    assert cell["num_ok"] == 4 and cell["output_tokens_total"] == 16
    assert cell["prefix_cache_reset"] is True
    assert cell["server_metrics"]["peaks"]["kv_cache_usage_max"] == 0.42
    assert "vllm:prompt_tokens_total" in cell["server_counters_delta"]
    assert data["workloads"]["chat"]["sha256"]
    assert "versions" in data["env"] and "git" in data["env"]
    assert data["quality"]["gsm8k"]["n"] == 1
    assert len(data["quality"]["agreement"]["outputs"]) == 6
    assert (sandbox / "L2" / "raw" / files[0].stem / "chat_c1.jsonl").exists()


def test_partial_run_resumes_and_skips_done_cells(sandbox, fake_server, monkeypatch):
    calls = []
    real = run.loadgen.run_closed_loop

    async def flaky(base_url, model, prompts, conc, params, client=None):
        if params.max_tokens == 4:  # a benchmark cell (not warmup/quality)
            calls.append(conc)
            if len(calls) == 2:
                raise RuntimeError("simulated spot preemption")
        return await real(base_url, model, prompts, conc, params, client)

    monkeypatch.setattr(run.loadgen, "run_closed_loop", flaky)
    args = [
        "--level",
        "L4",
        "--base-url",
        fake_server.base_url,
        "--workloads",
        "chat",
        "--concurrency",
        "1",
        "2",
        "--skip-quality",
    ]
    with pytest.raises(RuntimeError):
        run.main(args)
    (path,) = (sandbox / "L4").glob("*.json")
    partial = json.loads(path.read_text())
    assert partial["status"] == "partial" and len(partial["cells"]) == 1

    run.main(args)
    assert list((sandbox / "L4").glob("*.json")) == [path]  # same file, resumed
    done = json.loads(path.read_text())
    assert done["status"] == "complete"
    assert [c["concurrency"] for c in done["cells"]] == [1, 2]
    assert calls == [1, 2, 2]  # c=1 was not re-run


def test_quick_runs_go_to_smoke_dir(sandbox, fake_server):
    run.main(["--level", "L3", "--base-url", fake_server.base_url, "--quick", "--skip-quality"])
    assert list((sandbox / "_smoke" / "L3").glob("*.json"))
    assert not (sandbox / "L3").exists()


def test_report_from_saved_results(sandbox, fake_server, tmp_path, monkeypatch):
    for level in ("L2", "L4"):
        run.main(["--level", level, "--base-url", fake_server.base_url, "--concurrency", "1", "2"])
    runs = report.latest_complete(report.load_results(sandbox))
    assert list(runs) == ["L2", "L4"]

    rows = report.cell_rows(runs)
    assert len(rows) == 8 and all(r["source"].startswith("results/L") for r in rows)
    q = {r["level"]: r for r in report.quality_rows(runs)}
    assert q["L2"]["exact_match_rate"] is None  # L2 is the reference
    assert q["L4"]["exact_match_rate"] == 1.0  # fake server is deterministic

    plots = report.make_plots(rows, tmp_path / "plots")
    assert {p.name for p in plots} >= {"chat_throughput.png", "rag_ttft_p95.png"}
    md = report.summary_markdown(runs, [])
    assert "Workload: `chat`" in md and "| L4 |" in md

    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{report.START}\nold\n{report.END}\nrest\n")
    assert report.update_readme("NEW", readme)
    assert readme.read_text() == f"intro\n{report.START}\nNEW\n{report.END}\nrest\n"


def test_report_ignores_partial_smoke_and_variant_runs():
    base = {"status": "complete", "quick": False, "variant": None, "level": "L2"}
    runs = [
        {**base, "started_at": "2026-01-01"},
        {**base, "started_at": "2026-01-03", "status": "partial"},
        {**base, "started_at": "2026-01-04", "quick": True},
        {**base, "started_at": "2026-01-05", "variant": "x"},
        {**base, "started_at": "2026-01-02"},
    ]
    assert report.latest_complete(runs)["L2"]["started_at"] == "2026-01-02"
