import pytest
import yaml

from bench.config import LEVELS, ConfigError, load_all, load_level, validate


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
        "vllm": {},
    }
    (tmp_path / "L2.yaml").write_text(yaml.safe_dump(bad))
    with pytest.raises(ConfigError, match="does not match"):
        load_level("L2", configs_dir=tmp_path)
