"""Run the benchmark on a Modal L4: server and client in one container.

    modal run cloud/modal_app.py --level L2 --quick       # smoke test → results/_smoke/
    modal run cloud/modal_app.py --level L3               # one full level, then pull results
    modal run cloud/modal_app.py --level L5 --sweep
    modal run cloud/modal_app.py --level all --background # fire and forget; pull later
    modal run cloud/modal_app.py --pull                   # copy the results volume → results/

The repo (minus caches, results, .git) is shipped into the container and `bench.run` runs
there exactly as it would on a local GPU host. Server and load generator share the
container, so TTFT / ITL never include internet latency. The Hugging Face cache, vLLM's
compile cache and the results live on Modal Volumes, so weights download once and results
survive a crash or timeout. Billing is per second while the function runs; a function
can't run longer than Modal's 24 h limit.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import modal
from modal.volume import FileEntryType

REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTE_REPO = "/root/repo"
REMOTE_RESULTS = "/results"

APP_NAME = "qwen3-l4-bench"
GPU = "L4"
# Pinned here and recorded in every result file. Changing it means a new image build.
VLLM_VERSION = "0.31.0"
CUDA_IMAGE = "nvidia/cuda:12.9.0-devel-ubuntu22.04"
PYTHON = "3.12"
HOUR = 60 * 60

# Dockerignore-style patterns, relative to the repo root.
IGNORE = [
    ".git",
    ".venv",
    "venv",
    "hf_cache",
    "results",
    "plots",
    "notes",
    ".run",
    ".ruff_cache",
    ".pytest_cache",
    "**/__pycache__",
    "**/*.pyc",
    ".env",
    ".env.*",
]

app = modal.App(APP_NAME)

hf_cache = modal.Volume.from_name("qwen3-bench-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("qwen3-bench-vllm-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("qwen3-bench-results", create_if_missing=True)

image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python=PYTHON)
    .entrypoint([])
    .uv_pip_install(
        f"vllm=={VLLM_VERSION}",  # brings its own pinned torch + transformers
        "accelerate",
        "fastapi",
        "uvicorn[standard]",
        "httpx",
        "numpy",
        "matplotlib",
        "pyyaml",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",  # faster weight downloads
            "VLLM_LOG_STATS_INTERVAL": "1",
            "BENCH_RESULTS_DIR": REMOTE_RESULTS,
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_dir(REPO_ROOT, remote_path=REMOTE_REPO, ignore=IGNORE)
)


def bench_argv(
    level: str,
    quick: bool = False,
    sweep: bool = False,
    variant: str = "",
    skip_quality: bool = False,
    extra: str = "",
) -> list[str]:
    """Arguments for `python -m bench.run` (pure; unit-tested)."""
    argv = ["--level", level]
    if quick:
        argv.append("--quick")
    if sweep:
        argv.append("--sweep")
    if variant:
        argv += ["--variant", variant]
    if skip_quality:
        argv.append("--skip-quality")
    if extra:
        argv += shlex.split(extra)
    return argv


def read_hf_token(env_file: Path) -> str | None:
    """Only HF_TOKEN is forwarded from .env. HF_HOME etc. must not leak into the container."""
    if not env_file.exists():
        return None
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line.startswith("HF_TOKEN=") and not line.startswith("#"):
            value = line.split("=", 1)[1].strip().strip("'\"")
            return value or None
    return None


def _local_git() -> tuple[str | None, bool | None]:
    """Read-only git provenance from the local checkout (the container has no .git)."""

    def run(cmd: list[str]) -> str | None:
        try:
            return subprocess.run(
                cmd, capture_output=True, text=True, check=True, cwd=REPO_ROOT
            ).stdout.strip()
        except (subprocess.SubprocessError, OSError):
            return None

    sha = run(["git", "log", "-1", "--format=%H"])
    status = run(["git", "status", "--porcelain"])
    return sha, (bool(status) if status is not None else None)


def _secrets() -> list[modal.Secret]:
    token = read_hf_token(REPO_ROOT / ".env") if modal.is_local() else None
    return [modal.Secret.from_dict({"HF_TOKEN": token})] if token else []


@app.function(
    image=image,
    gpu=GPU,
    timeout=24 * HOUR,  # Modal's maximum; one level or sweep fits well within it
    cpu=4,
    memory=24 * 1024,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
        REMOTE_RESULTS: results_vol,
    },
    secrets=_secrets(),
)
def bench(argv: list[str], git_sha: str | None, git_dirty: bool | None) -> int:
    """Run `python -m bench.run <argv>` on the GPU; results land in the results volume."""
    env = dict(os.environ)
    env["BENCH_GIT_SHA"] = git_sha or ""
    if git_dirty is not None:
        env["BENCH_GIT_DIRTY"] = "1" if git_dirty else "0"
    subprocess.run(["nvidia-smi"], check=False)
    print(f"$ python -m bench.run {shlex.join(argv)}", flush=True)
    try:
        proc = subprocess.run([sys.executable, "-m", "bench.run", *argv], cwd=REMOTE_REPO, env=env)
        return proc.returncode
    finally:
        results_vol.commit()


def pull_results(dest: Path, include_raw: bool = False) -> list[Path]:
    """Copy result files from the volume into `dest`; skips files that are already identical."""
    copied: list[Path] = []
    for entry in results_vol.listdir("/", recursive=True):
        if entry.type != FileEntryType.FILE:
            continue
        rel = entry.path.lstrip("/")
        if not include_raw and "/raw/" in f"/{rel}":
            continue
        target = dest / rel
        data = b"".join(results_vol.read_file(entry.path))
        if target.exists() and target.read_bytes() == data:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        copied.append(target)
    return copied


@app.local_entrypoint()
def main(
    level: str = "L2",
    quick: bool = False,
    sweep: bool = False,
    variant: str = "",
    skip_quality: bool = False,
    extra: str = "",
    background: bool = False,
    pull: bool = False,
    include_raw: bool = False,
) -> None:
    dest = REPO_ROOT / "results"
    if pull:
        copied = pull_results(dest, include_raw)
        for p in copied:
            print(f"pulled {p.relative_to(REPO_ROOT)}")
        print(f"{len(copied)} file(s) updated under results/")
        return

    argv = bench_argv(level, quick, sweep, variant, skip_quality, extra)
    sha, dirty = _local_git()
    if background:
        call = bench.spawn(argv, sha, dirty)
        print(f"started {call.object_id}; follow with `modal app logs {APP_NAME}`")
        print("when it finishes: make pull-results")
        return

    rc = bench.remote(argv, sha, dirty)
    copied = pull_results(dest, include_raw)
    print(f"bench.run exit code {rc}; {len(copied)} result file(s) pulled into results/")
    if rc:
        sys.exit(rc)
