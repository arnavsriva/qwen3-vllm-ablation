import pytest
import yaml

from bench.config import (
    LEVELS,
    ConfigError,
    apply_overrides,
    config_hash,
    expand_sweep,
    load_all,
    load_bench,
    load_level,
    validate,
)


def test_every_level_has_a_valid_config():
    cfgs = load_all()
    assert list(cfgs) == list(LEVELS)


@pytest.mark.parametrize("level", LEVELS)
def test_level_field_matches_filename(level):
    assert load_level(level)["level"] == level


def test_only_l1_uses_hf():
    engines = {lvl: cfg["engine"] for lvl, cfg in load_all().items()}
    assert engines["L1"] == "hf"
    assert all(engines[lvl] == "vllm" for lvl in LEVELS[1:])


def test_levels_are_cumulative_on_the_key_switches():
    cfgs = load_all()
    prefix = {lvl: cfgs[lvl].get("vllm", {}).get("enable_prefix_caching") for lvl in LEVELS[1:]}
    assert prefix == {"L2": False, "L3": False, "L4": False, "L5": False, "L6": True, "L7": True}
    assert "attention_backend" not in cfgs["L2"]["vllm"]  # L2 = vLLM's own choice
    for lvl in LEVELS[2:]:
        assert cfgs[lvl]["vllm"]["attention_backend"] == "FLASH_ATTN"
    for lvl in LEVELS[3:]:
        assert cfgs[lvl]["vllm"]["quantization"] == "awq_marlin"
    assert cfgs["L7"]["vllm"]["speculative_config"]["method"] == "draft_model"
    # The deprecated env var must not come back: recent vLLM ignores it.
    assert all("VLLM_ATTENTION_BACKEND" not in (c.get("env") or {}) for c in cfgs.values())


def test_bench_workloads_cover_level_workloads():
    bench = load_bench()
    for cfg in load_all().values():
        assert set(cfg["workloads"]) <= set(bench["workloads"])
    assert bench["concurrency"] == [1, 4, 16, 64]


def test_unknown_level_rejected():
    with pytest.raises(ConfigError):
        load_level("L9")


def test_missing_keys_rejected():
    with pytest.raises(ConfigError, match="missing keys"):
        validate({"level": "L2"})


def test_mismatched_level_rejected(tmp_path):
    bad = {
        "level": "L3",
        "name": "x",
        "description": "x",
        "engine": "vllm",
        "model": "m",
        "common": {},
        "workloads": ["chat"],
        "vllm": {},
    }
    (tmp_path / "L2.yaml").write_text(yaml.safe_dump(bad))
    (tmp_path / "bench.yaml").write_text(yaml.safe_dump({"workloads": {"chat": {}}}))
    with pytest.raises(ConfigError, match="does not match"):
        load_level("L2", configs_dir=tmp_path)


def test_expand_sweep_grid():
    variants = expand_sweep(load_level("L5"))
    assert len(variants) == 9
    names = [n for n, _ in variants]
    assert "max_num_batched_tokens=2048,max_num_seqs=32" in names
    assert len(set(names)) == len(names)
    assert expand_sweep(load_level("L2")) == []


def test_apply_dotted_override_does_not_mutate():
    cfg = load_level("L7")
    out = apply_overrides(cfg, {"speculative_config.num_speculative_tokens": 6})
    assert out["vllm"]["speculative_config"]["num_speculative_tokens"] == 6
    assert out["vllm"]["speculative_config"]["model"] == "Qwen/Qwen3-0.6B"
    assert cfg["vllm"]["speculative_config"]["num_speculative_tokens"] == 4


def test_config_hash_is_stable_and_sensitive():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})
