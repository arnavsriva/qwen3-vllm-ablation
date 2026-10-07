"""Summary table + plots from saved results only.

    python -m bench.report                  # results/summary.{md,csv} + plots/*.png
    python -m bench.report --update-readme  # also refresh README between RESULTS markers

Per level it uses the newest *complete*, non-smoke, non-variant run. Every number written
comes from a file under results/, and the summary names that file.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any

from bench import quality
from bench.config import LEVELS, REPO_ROOT

# Overridable so a cloud container can write straight into a mounted volume.
RESULTS_DIR = Path(os.environ.get("BENCH_RESULTS_DIR") or REPO_ROOT / "results")
PLOTS_DIR = REPO_ROOT / "plots"
README = REPO_ROOT / "README.md"
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"

# Categorical slots in fixed order (validated: adjacent CVD ΔE ≥ 9.1, normal-vision ≥ 19.6
# on a light surface). Color follows the level, never its rank.
LEVEL_COLORS = dict(
    zip(
        LEVELS,
        ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"],
        strict=True,
    )
)
LEVEL_MARKERS = dict(zip(LEVELS, ["o", "s", "^", "D", "v", "p", "h"], strict=True))
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def load_results(results_dir: Path = RESULTS_DIR) -> list[dict[str, Any]]:
    runs = []
    for path in sorted(results_dir.glob("L*/*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        data["_path"] = str(path.relative_to(results_dir.parent))
        runs.append(data)
    return runs


def latest_complete(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Level → newest complete main (non-variant, non-quick) run."""
    best: dict[str, dict[str, Any]] = {}
    for r in runs:
        if r.get("status") != "complete" or r.get("quick") or r.get("variant"):
            continue
        cur = best.get(r["level"])
        if cur is None or r["started_at"] > cur["started_at"]:
            best[r["level"]] = r
    return dict(sorted(best.items()))


def _g(d: dict | None, *keys: str) -> Any:
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def cell_rows(runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for level, r in runs.items():
        for c in r["cells"]:
            rows.append(
                {
                    "level": level,
                    "workload": c["workload"],
                    "concurrency": c["concurrency"],
                    "output_tok_per_s": c["output_tok_per_s"],
                    "req_per_s": c["req_per_s"],
                    "ttft_p50_ms": _g(c, "ttft_ms", "p50"),
                    "ttft_p95_ms": _g(c, "ttft_ms", "p95"),
                    "itl_p50_ms": _g(c, "itl_ms", "p50"),
                    "itl_p95_ms": _g(c, "itl_ms", "p95"),
                    "gpu_mem_peak_mib": c.get("gpu_mem_peak_mib"),
                    "kv_cache_usage_max": _g(c, "server_metrics", "peaks", "kv_cache_usage_max"),
                    "num_ok": c["num_ok"],
                    "num_requests": c["num_requests"],
                    "source": r["_path"],
                }
            )
    return rows


def quality_rows(runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for level, r in runs.items():
        q = r.get("quality") or {}
        ref_level = _g(q, "agreement", "reference_level")
        ref = runs.get(ref_level) if ref_level else None
        agree = None
        if ref is not None and ref_level != level:
            agree = quality.agreement(
                _g(q, "agreement", "outputs") or {},
                _g(ref, "quality", "agreement", "outputs") or {},
            )
        rows.append(
            {
                "level": level,
                "gsm8k_accuracy": _g(q, "gsm8k", "accuracy"),
                "gsm8k_n": _g(q, "gsm8k", "n"),
                "agreement_vs": ref_level if agree else None,
                "exact_match_rate": agree["exact_match_rate"] if agree else None,
                "mean_prefix_fraction": agree["mean_prefix_fraction"] if agree else None,
                "source": r["_path"],
            }
        )
    return rows


def fmt(v: Any, nd: int = 1) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:,.{nd}f}"
    return str(v)


def provenance_rows(runs: dict[str, dict[str, Any]]) -> list[list[str]]:
    out = []
    for level, r in runs.items():
        env = r.get("env") or {}
        gpus = env.get("gpus") or [{}]
        out.append(
            [
                level,
                r["config"]["name"],
                gpus[0].get("name") or "–",
                env.get("driver_version") or "–",
                env.get("cuda_driver_api_version") or "–",
                _g(env, "versions", "vllm") or "–",
                r["started_at"][:19],
                f"`{r['_path']}`",
            ]
        )
    return out


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def summary_markdown(runs: dict[str, dict[str, Any]], plots: list[Path]) -> str:
    if not runs:
        return "_No complete runs in `results/` yet._"
    parts = []
    crows = cell_rows(runs)
    for wl in sorted({r["workload"] for r in crows}):
        sel = [r for r in crows if r["workload"] == wl]
        parts.append(f"#### Workload: `{wl}`\n")
        parts.append(
            md_table(
                [
                    "Level",
                    "Conc.",
                    "Out tok/s",
                    "Req/s",
                    "TTFT p50 / p95 (ms)",
                    "ITL p50 / p95 (ms)",
                    "Peak GPU mem (MiB)",
                    "OK",
                ],
                [
                    [
                        r["level"],
                        str(r["concurrency"]),
                        fmt(r["output_tok_per_s"]),
                        fmt(r["req_per_s"], 2),
                        f"{fmt(r['ttft_p50_ms'], 0)} / {fmt(r['ttft_p95_ms'], 0)}",
                        f"{fmt(r['itl_p50_ms'])} / {fmt(r['itl_p95_ms'])}",
                        fmt(r["gpu_mem_peak_mib"]),
                        f"{r['num_ok']}/{r['num_requests']}",
                    ]
                    for r in sel
                ],
            )
        )
        parts.append("")
    qrows = quality_rows(runs)
    parts.append("#### Quality\n")
    parts.append(
        md_table(
            ["Level", "GSM8K acc.", "Greedy exact-match vs ref", "Mean shared-prefix vs ref"],
            [
                [
                    r["level"],
                    f"{fmt(r['gsm8k_accuracy'], 3)} (n={fmt(r['gsm8k_n'])})",
                    f"{fmt(r['exact_match_rate'], 3)}"
                    + (f" (vs {r['agreement_vs']})" if r["agreement_vs"] else ""),
                    fmt(r["mean_prefix_fraction"], 3),
                ]
                for r in qrows
            ],
        )
    )
    parts.append("\n#### Provenance\n")
    parts.append(
        md_table(
            ["Level", "Name", "GPU", "Driver", "CUDA", "vLLM", "Started (UTC)", "Result file"],
            provenance_rows(runs),
        )
    )
    if plots:
        parts.append("")
        parts += [f"![{p.stem}]({p.relative_to(REPO_ROOT).as_posix()})" for p in plots]
    return "\n".join(parts)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def make_plots(rows: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    metrics = [
        ("output_tok_per_s", "Output tokens / s", "throughput"),
        ("ttft_p95_ms", "TTFT p95 (ms)", "ttft_p95"),
        ("itl_p50_ms", "Inter-token latency p50 (ms)", "itl_p50"),
    ]
    for wl in sorted({r["workload"] for r in rows}):
        sel = [r for r in rows if r["workload"] == wl]
        for key, label, slug in metrics:
            if all(r[key] is None for r in sel):
                continue
            fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
            fig.patch.set_facecolor(SURFACE)
            ax.set_facecolor(SURFACE)
            for level in LEVELS:
                pts = sorted(
                    (r["concurrency"], r[key])
                    for r in sel
                    if r["level"] == level and r[key] is not None
                )
                if not pts:
                    continue
                xs, ys = zip(*pts, strict=True)
                ax.plot(
                    xs,
                    ys,
                    color=LEVEL_COLORS[level],
                    marker=LEVEL_MARKERS[level],
                    linewidth=2,
                    markersize=6,
                    markeredgecolor=SURFACE,
                    markeredgewidth=1.5,
                    label=level,
                )
            ax.set_xscale("log", base=2)
            concs = sorted({r["concurrency"] for r in sel})
            ax.set_xticks(concs, [str(c) for c in concs])
            ax.set_xlabel("Concurrent requests", color=INK_2)
            ax.set_ylabel(label, color=INK_2)
            ax.set_title(f"{label} — {wl}", color=INK, loc="left", fontsize=11)
            ax.grid(True, color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            ax.tick_params(colors=INK_2)
            ax.legend(
                frameon=False,
                labelcolor=INK_2,
                ncol=1,
                loc="center left",
                bbox_to_anchor=(1.0, 0.5),
            )
            fig.tight_layout()
            path = out_dir / f"{wl}_{slug}.png"
            fig.savefig(path, facecolor=SURFACE)
            plt.close(fig)
            written.append(path)
    return written


def update_readme(section: str, readme: Path = README) -> bool:
    text = readme.read_text()
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
    if not pattern.search(text):
        return False
    new = pattern.sub(lambda _: f"{START}\n{section}\n{END}", text)
    readme.write_text(new)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--update-readme", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    runs = latest_complete(load_results())
    rows = cell_rows(runs)
    plots = [] if args.no_plots or not rows else make_plots(rows, PLOTS_DIR)
    md = summary_markdown(runs, plots)
    (RESULTS_DIR / "summary.md").write_text(md + "\n")
    write_csv(rows, RESULTS_DIR / "summary.csv")
    write_csv(quality_rows(runs), RESULTS_DIR / "quality.csv")
    print(f"{len(runs)} level(s) summarized → results/summary.md, {len(plots)} plot(s)")
    if args.update_readme:
        print("README updated" if update_readme(md) else f"README has no {START} markers")


if __name__ == "__main__":
    main()
