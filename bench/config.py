"""Load and validate per-level configs (configs/<level>.yaml) and shared bench settings."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
LEVELS = tuple(f"L{i}" for i in range(1, 8))
ENGINES = ("hf", "vllm")
REQUIRED_KEYS = ("level", "name", "description", "engine", "model", "common", "workloads")


class ConfigError(ValueError):
    pass


def load_bench(configs_dir: Path = CONFIGS_DIR) -> dict[str, Any]:
    with (configs_dir / "bench.yaml").open() as f:
        return yaml.safe_load(f)


def load_level(level: str, configs_dir: Path = CONFIGS_DIR) -> dict[str, Any]:
    """Return the parsed config for one level, raising ConfigError if it is malformed."""
    if level not in LEVELS:
        raise ConfigError(f"unknown level {level!r}; expected one of {LEVELS}")
    path = configs_dir / f"{level}.yaml"
    with path.open() as f:
        cfg = yaml.safe_load(f)
    known = set(load_bench(configs_dir)["workloads"])
    validate(cfg, expected_level=level, known_workloads=known)
    return cfg


def load_all(configs_dir: Path = CONFIGS_DIR) -> dict[str, dict[str, Any]]:
    return {level: load_level(level, configs_dir) for level in LEVELS}


def validate(
    cfg: Any, expected_level: str | None = None, known_workloads: set[str] | None = None
) -> None:
    if not isinstance(cfg, dict):
        raise ConfigError("config must be a mapping")
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ConfigError(f"missing keys: {missing}")
    if expected_level is not None and cfg["level"] != expected_level:
        raise ConfigError(f"level field {cfg['level']!r} does not match file {expected_level!r}")
    if cfg["engine"] not in ENGINES:
        raise ConfigError(f"engine must be one of {ENGINES}, got {cfg['engine']!r}")
    if cfg["engine"] == "vllm" and not isinstance(cfg.get("vllm"), dict):
        raise ConfigError("vllm engine requires a 'vllm' mapping of engine args")
    if cfg["engine"] == "hf" and not isinstance(cfg.get("hf"), dict):
        raise ConfigError("hf engine requires an 'hf' mapping")
    if known_workloads is not None:
        unknown = set(cfg["workloads"]) - known_workloads
        if unknown:
            raise ConfigError(f"unknown workloads: {sorted(unknown)}")
    if "sweep" in cfg and not cfg["sweep"].get("grid"):
        raise ConfigError("sweep section requires a non-empty 'grid'")


def set_dotted(d: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def apply_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of cfg with dotted-key overrides applied to its engine-args section."""
    out = copy.deepcopy(cfg)
    section = out.setdefault(cfg["engine"], {})
    for key, value in overrides.items():
        set_dotted(section, key, value)
    return out


def variant_name(overrides: dict[str, Any]) -> str:
    parts = []
    for key, value in overrides.items():
        short = key.split(".")[-1]
        parts.append(f"{short}={value}")
    return ",".join(parts)


def expand_sweep(cfg: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Cartesian product of the sweep grid → [(variant_name, overrides), ...]."""
    grid = cfg.get("sweep", {}).get("grid", {})
    if not grid:
        return []
    keys = list(grid)
    variants = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        overrides = dict(zip(keys, combo, strict=True))
        variants.append((variant_name(overrides), overrides))
    return variants


def config_hash(*parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]
