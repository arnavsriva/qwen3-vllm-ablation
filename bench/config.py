"""Load and validate per-level configs from configs/<level>.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
LEVELS = tuple(f"L{i}" for i in range(1, 8))
ENGINES = ("hf", "vllm")
REQUIRED_KEYS = ("level", "name", "description", "engine", "model", "common")


class ConfigError(ValueError):
    pass


def load_level(level: str, configs_dir: Path = CONFIGS_DIR) -> dict[str, Any]:
    """Return the parsed config for one level, raising ConfigError if it is malformed."""
    if level not in LEVELS:
        raise ConfigError(f"unknown level {level!r}; expected one of {LEVELS}")
    path = configs_dir / f"{level}.yaml"
    with path.open() as f:
        cfg = yaml.safe_load(f)
    validate(cfg, expected_level=level)
    return cfg


def load_all(configs_dir: Path = CONFIGS_DIR) -> dict[str, dict[str, Any]]:
    return {level: load_level(level, configs_dir) for level in LEVELS}


def validate(cfg: Any, expected_level: str | None = None) -> None:
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
