from bench.server_metrics import counter_deltas, parse_prometheus, peak_gauges

TEXT = """\
# HELP vllm:num_preemptions_total Cumulative preemptions
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total{engine="0",model_name="m"} 3.0
vllm:prompt_tokens_total{engine="0",model_name="m"} 100.0
vllm:prompt_tokens_total{engine="1",model_name="m"} 50.0
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.75
vllm:time_to_first_token_seconds_bucket{le="0.1"} 4.0
vllm:time_to_first_token_seconds_created 1.7e9
process_cpu_seconds_total 12.5
"""


def test_parse_sums_label_sets_and_skips_buckets():
    m = parse_prometheus(TEXT)
    assert m["vllm:prompt_tokens_total"] == 150.0
    assert m["vllm:kv_cache_usage_perc"] == 0.75
    assert not any(k.endswith(("_bucket", "_created")) for k in m)


def test_counter_deltas_only_vllm_totals():
    before = parse_prometheus(TEXT)
    after = dict(before, **{"vllm:prompt_tokens_total": 400.0, "process_cpu_seconds_total": 99.0})
    d = counter_deltas(before, after)
    assert d["vllm:prompt_tokens_total"] == 250.0
    assert d["vllm:num_preemptions_total"] == 0.0
    assert "process_cpu_seconds_total" not in d
    assert "vllm:kv_cache_usage_perc" not in d


def test_peak_gauges_tracks_max_and_old_name():
    peaks: dict[str, float] = {}
    peak_gauges({"vllm:kv_cache_usage_perc": 0.3}, peaks)
    peak_gauges({"vllm:gpu_cache_usage_perc": 0.6}, peaks)
    peak_gauges({"vllm:kv_cache_usage_perc": 0.5, "vllm:num_requests_waiting": 7}, peaks)
    assert peaks == {"kv_cache_usage_max": 0.6, "requests_waiting_max": 7}
