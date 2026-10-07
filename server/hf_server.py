"""L1 baseline: Hugging Face transformers behind a minimal OpenAI-compatible endpoint.

One request at a time (a global lock): no batching, no paged KV cache. This is the
reference the vLLM levels are measured against. It speaks the same streaming chat API as
vLLM so the benchmark client is identical across all levels.

GPU host only (needs torch, transformers, fastapi, uvicorn).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
from typing import Any

from server import openai_format as fmt


class TokenQueueStreamer:
    """transformers streamer that forwards each generated token id to an asyncio queue.

    generate() calls put() once with the prompt ids first, then once per new token.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue) -> None:
        self.loop = loop
        self.queue = queue
        self._seen_prompt = False

    def put(self, value: Any) -> None:
        if not self._seen_prompt:
            self._seen_prompt = True
            return
        for tok in value.reshape(-1).tolist():
            self.loop.call_soon_threadsafe(self.queue.put_nowait, int(tok))

    def end(self) -> None:
        self.loop.call_soon_threadsafe(self.queue.put_nowait, None)


def build_app(cfg: dict[str, Any]):
    import torch
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = cfg["model"]
    dtype = getattr(torch, cfg["common"]["dtype"])
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        attn_implementation=cfg["hf"].get("attn_implementation", "sdpa"),
        device_map="cuda",
    ).eval()
    torch.manual_seed(cfg["common"]["seed"])
    lock = asyncio.Lock()  # one request at a time: this level has no batching by design
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_id, "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        if not body.get("stream"):
            return JSONResponse({"error": "only stream=true is supported"}, status_code=400)
        tmpl_kwargs = body.get("chat_template_kwargs") or {}
        prompt_text = tokenizer.apply_chat_template(
            body["messages"], add_generation_prompt=True, tokenize=False, **tmpl_kwargs
        )
        prompt_ids = tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(model.device)
        max_new = int(body.get("max_tokens", 256))
        temperature = float(body.get("temperature", 0.0))
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        if body.get("ignore_eos"):
            gen_kwargs["min_new_tokens"] = max_new
        continuous = bool((body.get("stream_options") or {}).get("continuous_usage_stats"))
        n_prompt = int(prompt_ids.shape[-1])

        async def stream():
            cid = fmt.new_completion_id()
            async with lock:
                loop = asyncio.get_running_loop()
                queue: asyncio.Queue = asyncio.Queue()
                streamer = TokenQueueStreamer(loop, queue)

                def work():
                    with torch.inference_mode():
                        model.generate(prompt_ids, streamer=streamer, **gen_kwargs)

                thread = threading.Thread(target=work, daemon=True)
                thread.start()
                yield fmt.chunk(cid, model_id, role="assistant", content="")
                ids: list[int] = []
                emitted = ""
                while (tok := await queue.get()) is not None:
                    ids.append(tok)
                    text = tokenizer.decode(ids, skip_special_tokens=True)
                    delta, emitted = text[len(emitted) :], text
                    stats = fmt.usage(n_prompt, len(ids)) if continuous else None
                    # One chunk per token. A token that ends mid UTF-8 character decodes to ""
                    # here; its text arrives with the next token (token counts come from usage).
                    yield fmt.chunk(cid, model_id, content=delta, usage_stats=stats)
                thread.join()
                finish = "length" if len(ids) >= max_new else "stop"
                yield fmt.chunk(cid, model_id, finish_reason=finish)
                yield fmt.chunk(
                    cid, model_id, include_choice=False, usage_stats=fmt.usage(n_prompt, len(ids))
                )
                yield fmt.DONE

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="resolved level config (JSON)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)

    import uvicorn

    uvicorn.run(build_app(cfg), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
