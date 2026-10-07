"""Build the fixed workload + eval files from public datasets (seed 0, pinned revisions).

    pip install -e ".[workloads]" && python scripts/build_workloads.py

Outputs (committed; the benchmark never re-samples at run time):
  bench/workloads/chat.jsonl           256 UltraChat first-turn prompts, 32–512 tokens
  bench/workloads/rag.jsonl            64 SQuAD v2 questions + ~3–3.6k tokens of passages
  bench/workloads/shared_prefix.jsonl  128 SQuAD v2 questions sharing one ~2k-token system prompt
  bench/workloads/warmup.jsonl         8 short UltraChat prompts (not used in any cell)
  bench/eval/gsm8k_100.jsonl           100 GSM8K test questions with gold answers
  bench/eval/agreement_50.jsonl        50 UltraChat prompts for greedy-output agreement
  bench/workloads/MANIFEST.json        sources, dataset revisions, token stats, sha256
"""

from __future__ import annotations

import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
WL_DIR = ROOT / "bench" / "workloads"
EVAL_DIR = ROOT / "bench" / "eval"
SEED = 0
TOKENIZER = "Qwen/Qwen3-8B"

SOURCES = {
    "ultrachat": ("HuggingFaceH4/ultrachat_200k", None, "test_sft", "MIT"),
    "squad_v2": ("rajpurkar/squad_v2", None, "validation", "CC BY-SA 4.0"),
    "gsm8k": ("openai/gsm8k", "main", "test", "MIT"),
}

RAG_SYSTEM = (
    "You are a helpful assistant. Answer the question using only the provided passages. Be concise."
)
SHARED_SYSTEM_HEAD = (
    "You are a support assistant for a reference library. Answer every question using only "
    "the reference documents below. If the documents do not contain the answer, say so.\n\n"
    "REFERENCE DOCUMENTS\n"
)
GSM8K_SUFFIX = (
    "\n\nSolve the problem step by step, then finish with a final line of the form "
    "'The answer is N'."
)


def main() -> None:
    rng = random.Random(SEED)
    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    def ntok(s: str) -> int:
        return len(tok(s, add_special_tokens=False).input_ids)

    revisions, data = {}, {}
    api = HfApi()
    for key, (repo, config, split, _lic) in SOURCES.items():
        revisions[key] = api.dataset_info(repo).sha
        data[key] = load_dataset(repo, config, split=split, revision=revisions[key])

    # --- chat, agreement, warmup: disjoint UltraChat prompts --------------------------------
    uc = data["ultrachat"]
    order = list(range(len(uc)))
    rng.shuffle(order)
    chat, agree, warm = [], [], []
    seen: set[str] = set()
    for i in order:
        prompt = uc[i]["prompt"].strip()
        if prompt in seen:
            continue
        n = ntok(prompt)
        if len(warm) < 8 and 16 <= n <= 64:
            bucket = warm
        elif len(chat) < 256 and 32 <= n <= 512:
            bucket = chat
        elif len(agree) < 50 and 32 <= n <= 256:
            bucket = agree
        else:
            if len(chat) >= 256 and len(agree) >= 50 and len(warm) >= 8:
                break
            continue
        seen.add(prompt)
        bucket.append(
            {
                "id": f"uc-{uc[i]['prompt_id'][:12]}",
                "messages": [{"role": "user", "content": prompt}],
                "input_tokens": n,
            }
        )

    # --- RAG + shared prefix: SQuAD v2 ------------------------------------------------------
    sq = data["squad_v2"]
    contexts_by_title: dict[str, list[str]] = defaultdict(list)
    all_qs, answerable = [], []
    for row in sq:
        if row["context"] not in contexts_by_title[row["title"]]:
            contexts_by_title[row["title"]].append(row["context"])
        all_qs.append(row)
        if row["answers"]["text"]:
            answerable.append(row)
    titles = sorted(contexts_by_title)
    ctx_tokens = {c: ntok(c) for cs in contexts_by_title.values() for c in cs}

    # Shared prefix: documents from the first few shuffled titles, ~2k tokens.
    shuffled_titles = titles[:]
    rng.shuffle(shuffled_titles)
    shared_ctx, total, used_titles = [], 0, []
    for t in shuffled_titles:
        if total >= 2000:
            break
        used_titles.append(t)
        for c in contexts_by_title[t]:
            if total + ctx_tokens[c] > 2100:
                break
            shared_ctx.append(c)
            total += ctx_tokens[c]
    shared_set = set(shared_ctx)
    system_prompt = SHARED_SYSTEM_HEAD + "\n\n".join(
        f"[Doc {i + 1}] {c}" for i, c in enumerate(shared_ctx)
    )
    # Answerable *and* unanswerable SQuAD v2 questions about these documents: the system
    # prompt tells the model to say so when the documents don't contain the answer.
    shared_qs = [r for r in all_qs if r["context"] in shared_set]
    rng.shuffle(shared_qs)
    shared = [
        {
            "id": f"sp-{r['id']}",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": r["question"]},
            ],
            "input_tokens": ntok(system_prompt) + ntok(r["question"]),
            "answers": r["answers"]["text"],
        }
        for r in shared_qs[:128]
    ]

    # RAG: answerable questions outside the shared set, each with a distinct gold passage.
    # Distractors come only from *other* articles so the gold passage is the only source.
    rag_pool = [r for r in answerable if r["title"] not in set(used_titles)]
    rng.shuffle(rag_pool)
    rag, gold_used = [], set()
    rag_titles = [t for t in titles if t not in set(used_titles)]
    for r in rag_pool:
        if len(rag) >= 64:
            break
        if r["context"] in gold_used:
            continue
        gold_used.add(r["context"])
        others = [c for t in rag_titles if t != r["title"] for c in contexts_by_title[t]]
        passages, total = [r["context"]], ctx_tokens[r["context"]]
        while total < 3000:
            c = rng.choice(others)
            if c in passages or total + ctx_tokens[c] > 3600:
                continue
            passages.append(c)
            total += ctx_tokens[c]
        rng.shuffle(passages)
        body = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
        user = f"Passages:\n{body}\n\nQuestion: {r['question']}"
        rag.append(
            {
                "id": f"rag-{r['id']}",
                "messages": [
                    {"role": "system", "content": RAG_SYSTEM},
                    {"role": "user", "content": user},
                ],
                "input_tokens": ntok(RAG_SYSTEM) + ntok(user),
                "answers": r["answers"]["text"],
            }
        )

    # --- GSM8K ------------------------------------------------------------------------------
    gs = data["gsm8k"]
    idx = list(range(len(gs)))
    rng.shuffle(idx)
    gsm = []
    for i in idx[:100]:
        q = gs[i]["question"]
        gsm.append(
            {
                "id": f"gsm8k-{i}",
                "messages": [{"role": "user", "content": q + GSM8K_SUFFIX}],
                "answer": gs[i]["answer"].split("####")[-1].strip().replace(",", ""),
                "input_tokens": ntok(q + GSM8K_SUFFIX),
            }
        )

    outputs = {
        WL_DIR / "chat.jsonl": chat,
        WL_DIR / "rag.jsonl": rag,
        WL_DIR / "shared_prefix.jsonl": shared,
        WL_DIR / "warmup.jsonl": warm,
        EVAL_DIR / "gsm8k_100.jsonl": gsm,
        EVAL_DIR / "agreement_50.jsonl": agree,
    }
    manifest = {
        "seed": SEED,
        "tokenizer": TOKENIZER,
        "sources": {
            k: {
                "repo": v[0],
                "config": v[1],
                "split": v[2],
                "license": v[3],
                "revision": revisions[k],
            }
            for k, v in SOURCES.items()
        },
        "files": {},
    }
    for path, rows in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        path.write_text(text, encoding="utf-8")
        toks = [r["input_tokens"] for r in rows]
        stats = {"min": min(toks), "mean": round(statistics.mean(toks), 1), "max": max(toks)}
        rel = str(path.relative_to(ROOT))
        manifest["files"][rel] = {
            "rows": len(rows),
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "input_tokens": stats,
        }
        print(f"{rel}: {len(rows)} rows, input tokens {stats}")
    (WL_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
