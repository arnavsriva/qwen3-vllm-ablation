# Workload & eval sources

The benchmark inputs are fixed, committed files. They are built once by
`scripts/build_workloads.py` (seed 0, dataset revisions pinned in `MANIFEST.json`) and
never re-sampled at run time. `tests/test_workloads.py` fails if a file changes without
its manifest hash being updated.

| File | Source | License |
|---|---|---|
| `chat.jsonl`, `warmup.jsonl`, `../eval/agreement_50.jsonl` | [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) (`test_sft`, first user turn) | MIT |
| `rag.jsonl`, `shared_prefix.jsonl` | [rajpurkar/squad_v2](https://huggingface.co/datasets/rajpurkar/squad_v2) (`validation`, answerable questions; passages reused as retrieved context) | CC BY-SA 4.0 |
| `../eval/gsm8k_100.jsonl` | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) (`main`, `test`) | MIT |

- **rag:** one question per Wikipedia article. The gold passage is mixed with passages
  from other articles in shuffled order, about 3.0–3.6k tokens in total.
- **shared_prefix:** every request carries the same ~2k-token system prompt (reference
  documents) followed by a different question about those documents. This is the
  workload that tests prefix caching (L6).
