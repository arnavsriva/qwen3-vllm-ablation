"""Start / stop the inference server for one level.

    python -m server.launch --level L3                 # start in background, wait for health
    python -m server.launch --level L3 --foreground    # run attached (Ctrl-C to stop)
    python -m server.launch --stop

L1 runs server.hf_server; L2–L7 run `vllm serve` with flags generated from the config.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from bench.config import REPO_ROOT, apply_overrides, expand_sweep, load_bench, load_level

RUN_DIR = REPO_ROOT / ".run"
STATE_FILE = RUN_DIR / "server.json"

# Log lines worth keeping as evidence of what the server actually used (e.g. L3 must prove
# the FlashAttention backend was selected, not just requested).
EVIDENCE = re.compile(
    r"attention backend|_ATTN|flash ?attn|flashinfer|awq|marlin|quantiz|prefix cach|"
    r"chunked prefill|speculative|draft|max_num_seqs|max_num_batched_tokens|KV cache|"
    r"version|attn_implementation",
    re.IGNORECASE,
)


def _flag(key: str) -> str:
    return "--" + key.replace("_", "-")


def vllm_args(engine_args: dict[str, Any]) -> list[str]:
    """Map a config dict onto `vllm serve` CLI flags."""
    out: list[str] = []
    for key, value in engine_args.items():
        if value is None:
            continue
        if isinstance(value, bool):
            out.append(_flag(key) if value else "--no-" + key.replace("_", "-"))
        elif isinstance(value, dict):
            out += [_flag(key), json.dumps(value, sort_keys=True)]
        elif isinstance(value, list):
            out += [_flag(key), *map(str, value)]
        else:
            out += [_flag(key), str(value)]
    return out


def build_command(
    cfg: dict[str, Any], host: str, port: int, config_path: Path
) -> tuple[list[str], dict[str, str]]:
    """Return (argv, extra env) for the level's server process."""
    env = {str(k): str(v) for k, v in (cfg.get("env") or {}).items()}
    if cfg["engine"] == "hf":
        argv = [sys.executable, "-m", "server.hf_server", "--config", str(config_path)]
        argv += ["--host", host, "--port", str(port)]
        return argv, env

    common = cfg["common"]
    vllm_bin = str(Path(sys.executable).parent / "vllm")
    argv = [vllm_bin, "serve", cfg["model"], "--host", host, "--port", str(port)]
    argv += ["--dtype", str(common["dtype"]), "--max-model-len", str(common["max_model_len"])]
    argv += ["--seed", str(common["seed"])]
    argv += vllm_args(cfg["vllm"])
    # Enables POST /reset_prefix_cache so cells can't warm each other's cache.
    env["VLLM_SERVER_DEV_MODE"] = "1"
    return argv, env


@dataclass
class ServerHandle:
    pid: int
    base_url: str
    log_path: Path
    argv: list[str]
    env: dict[str, str]
    proc: subprocess.Popen | None = None

    def alive(self) -> bool:
        if self.proc is not None:
            return self.proc.poll() is None
        try:
            os.kill(self.pid, 0)
        except OSError:
            return False
        return True


def wait_healthy(handle: ServerHandle, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not handle.alive():
            raise RuntimeError(
                f"server exited during startup; see {handle.log_path}\n{tail(handle.log_path)}"
            )
        try:
            if httpx.get(f"{handle.base_url}/health", timeout=5).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(3)
    raise TimeoutError(f"server not healthy after {timeout_s}s; see {handle.log_path}")


def tail(path: Path, n: int = 30) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def log_evidence(path: Path, limit: int = 60) -> list[str]:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return []
    return [ln.strip()[:400] for ln in lines if EVIDENCE.search(ln)][:limit]


def start(
    cfg: dict[str, Any], *, host: str, port: int, log_path: Path, timeout_s: float
) -> ServerHandle:
    if STATE_FILE.exists():
        stop()
    RUN_DIR.mkdir(exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    config_path = RUN_DIR / "server_config.json"
    config_path.write_text(json.dumps(cfg, indent=2))
    argv, extra_env = build_command(cfg, host, port, config_path)
    log = log_path.open("w")
    proc = subprocess.Popen(
        argv,
        cwd=REPO_ROOT,
        env={**os.environ, **extra_env},
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # own process group, so stop() kills workers too
    )
    handle = ServerHandle(proc.pid, f"http://{host}:{port}", log_path, argv, extra_env, proc)
    STATE_FILE.write_text(
        json.dumps({"pid": proc.pid, "base_url": handle.base_url, "log": str(log_path)})
    )
    try:
        wait_healthy(handle, timeout_s)
    except BaseException:
        stop()
        raise
    return handle


def stop(grace_s: float = 30.0) -> bool:
    """Stop the server recorded in .run/server.json. Returns True if one was running."""
    if not STATE_FILE.exists():
        return False
    pid = json.loads(STATE_FILE.read_text())["pid"]
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGTERM)
        deadline = time.monotonic() + grace_s
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except OSError:
                break
            time.sleep(0.5)
        else:
            os.killpg(pgid, signal.SIGKILL)
        was_running = True
    except ProcessLookupError:
        was_running = False
    STATE_FILE.unlink(missing_ok=True)
    return was_running


def resolve_config(level: str, variant: str | None) -> dict[str, Any]:
    cfg = load_level(level)
    if variant:
        variants = dict(expand_sweep(cfg))
        if variant not in variants:
            raise SystemExit(f"unknown variant {variant!r} for {level}; have {sorted(variants)}")
        cfg = apply_overrides(cfg, variants[variant])
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--level")
    ap.add_argument("--variant")
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--foreground", action="store_true")
    ap.add_argument("--print-command", action="store_true")
    args = ap.parse_args()

    if args.stop:
        print("stopped" if stop() else "no server running")
        return
    if not args.level:
        ap.error("--level is required")

    bench = load_bench()
    cfg = resolve_config(args.level, args.variant)
    RUN_DIR.mkdir(exist_ok=True)
    config_path = RUN_DIR / "server_config.json"
    config_path.write_text(json.dumps(cfg, indent=2))
    argv, extra_env = build_command(cfg, bench["host"], bench["port"], config_path)
    if args.print_command:
        print(" ".join(f"{k}={v}" for k, v in extra_env.items()), " ".join(argv))
        return
    if args.foreground:
        os.execvpe(argv[0], argv, {**os.environ, **extra_env})

    log_path = REPO_ROOT / "logs" / f"server-{args.level}.log"
    handle = start(
        cfg,
        host=bench["host"],
        port=bench["port"],
        log_path=log_path,
        timeout_s=bench["server_start_timeout_s"],
    )
    print(f"{args.level} healthy at {handle.base_url} (pid {handle.pid}, log {log_path})")


if __name__ == "__main__":
    main()
