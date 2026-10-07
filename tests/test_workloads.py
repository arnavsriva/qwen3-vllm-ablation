"""The committed workload/eval files are the benchmark's fixed inputs: guard them."""

import json

import pytest

from bench.config import REPO_ROOT, load_bench
from bench.workloads import load_jsonl, sha256_file

BENCH = load_bench()
MANIFEST = json.loads((REPO_ROOT / "bench/workloads/MANIFEST.json").read_text())


@pytest.mark.parametrize("path", sorted(MANIFEST["files"]))
def test_manifest_hash_matches_file(path):
    assert sha256_file(path) == MANIFEST["files"][path]["sha256"], (
        f"{path} changed without rebuilding MANIFEST (run `make workloads`)"
    )


@pytest.mark.parametrize("name", sorted(BENCH["workloads"]))
def test_workload_has_enough_unique_rows(name):
    rows = load_jsonl(BENCH["workloads"][name]["file"])
    assert len(rows) >= BENCH["workloads"][name]["num_requests"]
    assert len({r["id"] for r in rows}) == len(rows)


def test_prompts_fit_context_window():
    max_len = 8192 - BENCH["generation"]["max_tokens"]
    for name, wl in BENCH["workloads"].items():
        for r in load_jsonl(wl["file"]):
            # Raw content tokens; the chat template adds a few dozen more.
            assert r["input_tokens"] < max_len - 100, (name, r["id"])


def test_shared_prefix_rows_share_one_system_prompt():
    rows = load_jsonl(BENCH["workloads"]["shared_prefix"]["file"])
    systems = {r["messages"][0]["content"] for r in rows}
    assert len(systems) == 1
    assert all(r["messages"][0]["role"] == "system" for r in rows)


def test_rag_is_long_context():
    rows = load_jsonl(BENCH["workloads"]["rag"]["file"])
    assert min(r["input_tokens"] for r in rows) >= 3000


def test_eval_sets_and_disjointness():
    gsm = load_jsonl(BENCH["quality"]["gsm8k"]["file"])
    assert len(gsm) == 100
    assert all(r["answer"].lstrip("-").replace(".", "").isdigit() for r in gsm)
    agree = load_jsonl(BENCH["quality"]["agreement"]["file"])
    assert len(agree) == 50
    chat = {r["id"] for r in load_jsonl(BENCH["workloads"]["chat"]["file"])}
    warm = {r["id"] for r in load_jsonl(BENCH["warmup_file"])}
    assert not chat & {r["id"] for r in agree}
    assert not chat & warm
