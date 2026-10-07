import json
from pathlib import Path

from bench.config import apply_overrides, load_level
from server.launch import build_command, log_evidence, vllm_args


def argv_for(level, overrides=None):
    cfg = load_level(level)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    return build_command(cfg, "127.0.0.1", 8000, Path("/tmp/cfg.json"))


def flag_value(argv, flag):
    return argv[argv.index(flag) + 1]


def test_vllm_args_mapping():
    args = vllm_args(
        {"enable_prefix_caching": False, "max_num_seqs": 64, "x": None, "spec": {"b": 1, "a": 2}}
    )
    assert args == [
        "--no-enable-prefix-caching",
        "--max-num-seqs",
        "64",
        "--spec",
        '{"a": 2, "b": 1}',
    ]
    assert vllm_args({"enable_chunked_prefill": True}) == ["--enable-chunked-prefill"]


def test_l1_runs_hf_server():
    argv, env = argv_for("L1")
    assert argv[1:3] == ["-m", "server.hf_server"]
    assert flag_value(argv, "--config") == "/tmp/cfg.json"
    assert "VLLM_SERVER_DEV_MODE" not in env


def test_l2_is_vllm_with_prefix_caching_off():
    argv, env = argv_for("L2")
    assert argv[1:3] == ["serve", "Qwen/Qwen3-8B"]
    assert "--no-enable-prefix-caching" in argv
    assert "--attention-backend" not in argv
    assert flag_value(argv, "--dtype") == "bfloat16"
    assert flag_value(argv, "--max-model-len") == "8192"
    assert env["VLLM_SERVER_DEV_MODE"] == "1"


def test_l3_pins_flash_attention_via_cli_flag():
    argv, env = argv_for("L3")
    assert flag_value(argv, "--attention-backend") == "FLASH_ATTN"
    assert "VLLM_ATTENTION_BACKEND" not in env


def test_l3_variant_swaps_backend():
    argv, _ = argv_for("L3", {"attention_backend": "TRITON_ATTN"})
    assert flag_value(argv, "--attention-backend") == "TRITON_ATTN"


def test_l4_awq():
    argv, _ = argv_for("L4")
    assert argv[2] == "Qwen/Qwen3-8B-AWQ"
    assert flag_value(argv, "--quantization") == "awq_marlin"


def test_l6_prefix_caching_on():
    argv, _ = argv_for("L6")
    assert "--enable-prefix-caching" in argv


def test_l7_speculative_config_is_json():
    argv, _ = argv_for("L7")
    spec = json.loads(flag_value(argv, "--speculative-config"))
    assert spec == {
        "method": "draft_model",
        "model": "Qwen/Qwen3-0.6B",
        "num_speculative_tokens": 4,
    }


def test_log_evidence_filters(tmp_path):
    log = tmp_path / "server.log"
    log.write_text(
        "INFO starting\nINFO Using FLASH_ATTN attention backend\nDEBUG noise\n"
        "INFO Prefix caching is disabled\n"
    )
    lines = log_evidence(log)
    assert lines == ["INFO Using FLASH_ATTN attention backend", "INFO Prefix caching is disabled"]
