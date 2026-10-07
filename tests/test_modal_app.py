"""CPU-only checks for the Modal launcher: argument building, .env filtering, provenance.

Skipped when the `modal` client isn't installed (CI installs only dev+bench).
"""

import importlib

import pytest

from bench import env as bench_env

modal = pytest.importorskip("modal")
modal_app = importlib.import_module("cloud.modal_app")


def test_bench_argv_minimal():
    assert modal_app.bench_argv("L2") == ["--level", "L2"]


def test_bench_argv_all_flags():
    argv = modal_app.bench_argv(
        "L5", quick=True, sweep=True, variant="v1", skip_quality=True, extra="--concurrency 1 4"
    )
    assert argv == [
        "--level",
        "L5",
        "--quick",
        "--sweep",
        "--variant",
        "v1",
        "--skip-quality",
        "--concurrency",
        "1",
        "4",
    ]


def test_read_hf_token_only_forwards_token(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("HF_HOME=./hf_cache\n# HF_TOKEN=commented\nHF_TOKEN='hf_abc'\n")
    assert modal_app.read_hf_token(env_file) == "hf_abc"
    env_file.write_text("HF_TOKEN=\n")
    assert modal_app.read_hf_token(env_file) is None
    assert modal_app.read_hf_token(tmp_path / "missing") is None


def test_ignore_excludes_heavy_and_private_dirs():
    for name in (".git", ".venv", "hf_cache", "results", "notes", ".env"):
        assert name in modal_app.IGNORE


def test_git_info_from_env(monkeypatch):
    monkeypatch.setenv("BENCH_GIT_SHA", "abc123")
    monkeypatch.setenv("BENCH_GIT_DIRTY", "1")
    assert bench_env.git_info() == {"sha": "abc123", "dirty": True, "source": "env"}
    monkeypatch.setenv("BENCH_GIT_SHA", "")
    monkeypatch.delenv("BENCH_GIT_DIRTY")
    assert bench_env.git_info() == {"sha": None, "dirty": None, "source": "env"}


def test_results_dir_override(monkeypatch, tmp_path):
    monkeypatch.setenv("BENCH_RESULTS_DIR", str(tmp_path))
    from bench import report, run

    importlib.reload(run)
    importlib.reload(report)
    try:
        assert tmp_path == run.RESULTS_DIR
        assert tmp_path == report.RESULTS_DIR
    finally:
        monkeypatch.delenv("BENCH_RESULTS_DIR")
        importlib.reload(run)
        importlib.reload(report)
