"""Benchmark one level, all levels, or a level's sweep variants.

    python -m bench.run --level L3                # start server, run all cells + quality, stop
    python -m bench.run --level all               # L1..L7 sequentially
    python -m bench.run --level L5 --sweep        # every variant in L5's sweep grid
    python -m bench.run --level L2 --quick        # smoke test → results/_smoke/ (git-ignored)
    python -m bench.run --level L2 --base-url http://127.0.0.1:8000   # use a running server

Writes results/<level>/<timestamp>[__<variant>].json after every cell, so a spot
preemption loses at most one cell: rerunning the same command resumes the partial file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from bench import client as loadgen
from bench import quality, workloads
from bench.config import (
    LEVELS,
    REPO_ROOT,
    apply_overrides,
    config_hash,
    expand_sweep,
    load_bench,
    load_level,
)
from bench.env import collect_env
from bench.gpu import GpuMemorySampler
from bench.metrics import summarize_cell
from bench.server_metrics import MetricsSampler
from server import launch

# Overridable so a cloud container can write straight into a mounted volume.
RESULTS_DIR = Path(os.environ.get("BENCH_RESULTS_DIR") or REPO_ROOT / "results")
SCHEMA_VERSION = 1


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    os.replace(tmp, path)


def find_resumable(out_dir: Path, chash: str) -> Path | None:
    for path in sorted(out_dir.glob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("status") == "partial" and data.get("config_hash") == chash:
            return path
    return None


async def reset_prefix_cache(base_url: str) -> bool | None:
    """True if reset, None if the server has no such endpoint (HF server)."""
    async with httpx.AsyncClient() as c:
        try:
            r = await c.post(f"{base_url}/reset_prefix_cache", timeout=30)
        except httpx.HTTPError:
            return None
    return True if r.status_code == 200 else None


def plan_run(cfg: dict[str, Any], bench: dict[str, Any], args: argparse.Namespace) -> dict:
    """Decide workloads, concurrency levels and request counts for this run."""
    sweep = cfg.get("sweep", {}) if (args.sweep or args.variant) else {}
    wl_names = args.workloads or sweep.get("workloads") or cfg["workloads"]
    conc = args.concurrency or sweep.get("concurrency") or bench["concurrency"]
    overrides = (cfg.get("bench_overrides") or {}).get("num_requests", {})
    counts = {}
    for name in wl_names:
        n = overrides.get(name, bench["workloads"][name]["num_requests"])
        if args.num_requests:
            n = args.num_requests
        elif args.quick:
            n = min(n, 8)
        counts[name] = n
    if args.quick and not args.concurrency:
        conc = [c for c in conc if c <= 4]
    skip_quality = args.skip_quality or bool(sweep.get("skip_quality"))
    return {"workloads": counts, "concurrency": list(conc), "skip_quality": skip_quality}


async def run_quality(
    base_url: str, model: str, bench: dict[str, Any], quick: bool, params_seed: int
) -> dict[str, Any]:
    q = bench["quality"]
    out: dict[str, Any] = {}

    rows = workloads.load_jsonl(q["gsm8k"]["file"], limit=10 if quick else None)
    params = loadgen.GenParams(
        max_tokens=q["gsm8k"]["max_tokens"], temperature=0.0, seed=params_seed
    )
    res, _ = await loadgen.run_closed_loop(base_url, model, rows, q["gsm8k"]["concurrency"], params)
    outputs = {r.id: r.text for r in res if r.ok}
    out["gsm8k"] = {
        **quality.gsm8k_score(rows, outputs),
        "source": workloads.describe(q["gsm8k"]["file"], len(rows)),
        "outputs": outputs,
    }

    rows = workloads.load_jsonl(q["agreement"]["file"], limit=5 if quick else None)
    params = loadgen.GenParams(
        max_tokens=q["agreement"]["max_tokens"], temperature=0.0, seed=params_seed
    )
    res, _ = await loadgen.run_closed_loop(
        base_url, model, rows, q["agreement"]["concurrency"], params
    )
    out["agreement"] = {
        "reference_level": q["agreement"]["reference_level"],
        "source": workloads.describe(q["agreement"]["file"], len(rows)),
        "outputs": {r.id: r.text for r in res if r.ok},
        "num_errors": sum(not r.ok for r in res),
    }
    return out


async def run_cells(
    data: dict[str, Any],
    save: Any,
    base_url: str,
    model: str,
    bench: dict[str, Any],
    plan: dict[str, Any],
    raw_dir: Path,
) -> None:
    g = bench["generation"]
    params = loadgen.GenParams(
        max_tokens=g["max_tokens"],
        temperature=g["temperature"],
        ignore_eos=g["ignore_eos"],
        enable_thinking=g["enable_thinking"],
        seed=bench["seed"],
    )
    done = {(c["workload"], c["concurrency"]) for c in data["cells"]}
    for wl_name, n in plan["workloads"].items():
        wl_file = bench["workloads"][wl_name]["file"]
        prompts = workloads.load_jsonl(wl_file, limit=n)
        data["workloads"][wl_name] = workloads.describe(wl_file, len(prompts))
        for conc in plan["concurrency"]:
            if (wl_name, conc) in done:
                print(f"  skip {wl_name} c={conc} (already in partial result)")
                continue
            print(f"  {wl_name} c={conc} n={len(prompts)} ...", flush=True)
            reset = await reset_prefix_cache(base_url)
            with GpuMemorySampler() as gpu:
                async with MetricsSampler(base_url) as sm:
                    results, duration = await loadgen.run_closed_loop(
                        base_url, model, prompts, conc, params
                    )
            cell = {
                "workload": wl_name,
                "concurrency": conc,
                **summarize_cell(results, duration),
                "gpu_mem_peak_mib": gpu.peak_mib,
                "prefix_cache_reset": reset,
                "server_metrics": {"available": sm.available, "peaks": sm.peaks},
                "server_counters_delta": sm.deltas,
            }
            data["cells"].append(cell)
            raw = raw_dir / f"{wl_name}_c{conc}.jsonl"
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_text("\n".join(json.dumps(r.record()) for r in results) + "\n")
            save()
            tok_s = cell["output_tok_per_s"] or 0
            ttft = (cell["ttft_ms"] or {}).get("p50", float("nan"))
            print(
                f"    ok={cell['num_ok']}/{cell['num_requests']} "
                f"{tok_s:.1f} tok/s  ttft_p50={ttft:.0f} ms"
            )


def run_one(
    level: str, variant: str | None, cfg: dict[str, Any], bench: dict, args: argparse.Namespace
) -> Path:
    plan = plan_run(cfg, bench, args)
    wl_hashes = {
        name: workloads.sha256_file(bench["workloads"][name]["file"]) for name in plan["workloads"]
    }
    chash = config_hash(cfg, bench, plan, wl_hashes, args.quick)
    out_dir = RESULTS_DIR / ("_smoke" if args.quick else "") / level
    path = find_resumable(out_dir, chash)
    if path:
        data = json.loads(path.read_text())
        print(f"[{level}{'/' + variant if variant else ''}] resuming {rel(path)}")
    else:
        stem = now_stamp() + (f"__{safe(variant)}" if variant else "")
        path = out_dir / f"{stem}.json"
        data = {
            "schema_version": SCHEMA_VERSION,
            "level": level,
            "variant": variant,
            "quick": args.quick,
            "status": "partial",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "config_hash": chash,
            "config": cfg,
            "bench": bench,
            "plan": plan,
            "workloads": {},
            "cells": [],
            "quality": None,
        }
        print(f"[{level}{'/' + variant if variant else ''}] → {rel(path)}")
    raw_dir = out_dir / "raw" / path.stem

    def save() -> None:
        save_json(path, data)

    handle = None
    base_url = args.base_url
    try:
        if not base_url:
            log_path = raw_dir / "server.log"
            handle = launch.start(
                cfg,
                host=bench["host"],
                port=bench["port"],
                log_path=log_path,
                timeout_s=bench["server_start_timeout_s"],
            )
            base_url = handle.base_url
            data["server"] = {
                "argv": handle.argv,
                "env": handle.env,
                "log_evidence": launch.log_evidence(log_path),
            }
        data["env"] = collect_env()
        save()

        async def go() -> None:
            warm = workloads.load_jsonl(bench["warmup_file"], limit=bench["warmup_requests"])
            await loadgen.run_closed_loop(
                base_url, cfg["model"], warm, 1, loadgen.GenParams(max_tokens=32)
            )
            await run_cells(data, save, base_url, cfg["model"], bench, plan, raw_dir)
            if not plan["skip_quality"] and data.get("quality") is None:
                print("  quality eval ...", flush=True)
                data["quality"] = await run_quality(
                    base_url, cfg["model"], bench, args.quick, bench["seed"]
                )
                acc = data["quality"]["gsm8k"]["accuracy"]
                print(f"    gsm8k accuracy={acc}")
                save()

        asyncio.run(go())
        if handle:  # re-read: the log keeps growing after startup
            data["server"]["log_evidence"] = launch.log_evidence(handle.log_path)
        data["status"] = "complete"
        data["finished_at"] = datetime.now(timezone.utc).isoformat()
        save()
    finally:
        if handle:
            launch.stop()
    return path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--level", required=True, help="L1..L7 or 'all'")
    ap.add_argument("--sweep", action="store_true", help="run every variant in the sweep grid")
    ap.add_argument("--variant", help="run one named sweep variant")
    ap.add_argument("--workloads", nargs="+")
    ap.add_argument("--concurrency", nargs="+", type=int)
    ap.add_argument("--num-requests", type=int)
    ap.add_argument("--quick", action="store_true", help="tiny smoke run → results/_smoke/")
    ap.add_argument("--skip-quality", action="store_true")
    ap.add_argument("--base-url", help="benchmark an already-running server")
    args = ap.parse_args(argv)

    bench = load_bench()
    levels = LEVELS if args.level == "all" else (args.level,)
    if args.base_url and len(levels) > 1:
        ap.error("--base-url works with a single level")
    for level in levels:
        cfg = load_level(level)
        if args.sweep or args.variant:
            variants = expand_sweep(cfg)
            if not variants:
                print(f"[{level}] no sweep grid; skipping", file=sys.stderr)
                continue
            if args.variant:
                variants = [(n, o) for n, o in variants if n == args.variant]
                if not variants:
                    ap.error(f"unknown variant {args.variant!r}")
            for name, overrides in variants:
                run_one(level, name, apply_overrides(cfg, overrides), bench, args)
        else:
            run_one(level, None, cfg, bench, args)


if __name__ == "__main__":
    main()
