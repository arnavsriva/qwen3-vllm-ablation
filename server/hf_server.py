"""L1 baseline: Hugging Face transformers behind a minimal OpenAI-compatible endpoint.

One request at a time (a global lock): no batching, no paged KV cache. This is the
reference the vLLM levels are measured against. It speaks the same streaming chat API as
vLLM so the benchmark client is identical across all levels.

The HTTP layer (`make_app`) only needs fastapi and is unit-tested on CPU with a stub
backend. The model layer (`load_backend`) needs torch + transformers and runs on the GPU
host only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

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


@dataclass
class Backend:
    """What the HTTP layer needs from a model. Real one in load_backend(); tests stub it."""

    model_id: str
    # (messages, chat_template_kwargs) -> (prompt_ids, num_prompt_tokens)
    encode: Callable[[list[dict[str, Any]], dict[str, Any]], tuple[Any, int]]
    # token ids -> text
    decode: Callable[[list[int]], str]
    # (prompt_ids, streamer, generate kwargs) -> None; blocking; must call streamer.put per token
    generate: Callable[[Any, TokenQueueStreamer, dict[str, Any]], None]


def load_backend(cfg: dict[str, Any]) -> Backend:
    """GPU host only. Loads the model and prints evidence lines for the result file."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = cfg["model"]
    dtype = getattr(torch, cfg["common"]["dtype"])
    attn = cfg["hf"].get("attn_implementation", "sdpa")
    t0 = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, attn_implementation=attn, device_map="cuda"
    ).eval()
    torch.manual_seed(cfg["common"]["seed"])
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    # These lines match server.launch.EVIDENCE and end up in results/*.json.
    print(
        f"[hf_server] model={model_id} dtype={cfg['common']['dtype']} "
        f"attn_implementation={model.config._attn_implementation} device={device} "
        f"load_s={time.monotonic() - t0:.1f}",
        flush=True,
    )
    print(
        f"[hf_server] torch version {torch.__version__} (cuda {torch.version.cuda}), "
        f"transformers version {transformers.__version__}",
        flush=True,
    )

    def encode(messages: list[dict[str, Any]], tmpl_kwargs: dict[str, Any]) -> tuple[Any, int]:
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, **tmpl_kwargs
        )
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
        return ids.to(model.device), int(ids.shape[-1])

    def decode(ids: list[int]) -> str:
        return tokenizer.decode(ids, skip_special_tokens=True)

    def generate(prompt_ids: Any, streamer: TokenQueueStreamer, kw: dict[str, Any]) -> None:
        with torch.inference_mode():
            model.generate(prompt_ids, streamer=streamer, pad_token_id=tokenizer.eos_token_id, **kw)

    return Backend(model_id, encode, decode, generate)


def gen_kwargs_from_body(body: dict[str, Any]) -> dict[str, Any]:
    """Map the OpenAI-style request onto transformers.generate() arguments."""
    max_new = int(body.get("max_tokens", 256))
    temperature = float(body.get("temperature", 0.0))
    kw: dict[str, Any] = {"max_new_tokens": max_new, "do_sample": temperature > 0}
    if temperature > 0:
        kw["temperature"] = temperature
    if body.get("ignore_eos"):
        kw["min_new_tokens"] = max_new  # same "exactly max_tokens" semantics as vLLM
    return kw


def make_app(backend: Backend) -> FastAPI:
    lock = asyncio.Lock()  # one request at a time: this level has no batching by design
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": backend.model_id, "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        if not body.get("stream"):
            return JSONResponse({"error": "only stream=true is supported"}, status_code=400)
        prompt_ids, n_prompt = backend.encode(
            body["messages"], body.get("chat_template_kwargs") or {}
        )
        kw = gen_kwargs_from_body(body)
        max_new = kw["max_new_tokens"]
        continuous = bool((body.get("stream_options") or {}).get("continuous_usage_stats"))

        async def stream():
            cid = fmt.new_completion_id()
            async with lock:
                loop = asyncio.get_running_loop()
                queue: asyncio.Queue = asyncio.Queue()
                streamer = TokenQueueStreamer(loop, queue)
                failure: list[BaseException] = []

                def work() -> None:
                    try:
                        backend.generate(prompt_ids, streamer, kw)
                    except BaseException as e:  # noqa: BLE001 - surfaced to the client below
                        failure.append(e)
                        streamer.end()  # unblock the reader; a second None is harmless

                thread = threading.Thread(target=work, daemon=True)
                thread.start()
                yield fmt.chunk(cid, backend.model_id, role="assistant", content="")
                ids: list[int] = []
                emitted = ""
                while (tok := await queue.get()) is not None:
                    ids.append(tok)
                    text = backend.decode(ids)
                    delta, emitted = text[len(emitted) :], text
                    stats = fmt.usage(n_prompt, len(ids)) if continuous else None
                    # One chunk per token. A token that ends mid UTF-8 character decodes to ""
                    # here; its text arrives with the next token (token counts come from usage).
                    yield fmt.chunk(cid, backend.model_id, content=delta, usage_stats=stats)
                thread.join()
                if failure:
                    print(f"[hf_server] generate failed: {failure[0]!r}", flush=True)
                    raise failure[0]  # drops the connection mid-stream → client counts an error
                finish = "length" if len(ids) >= max_new else "stop"
                yield fmt.chunk(cid, backend.model_id, finish_reason=finish)
                yield fmt.chunk(
                    cid,
                    backend.model_id,
                    include_choice=False,
                    usage_stats=fmt.usage(n_prompt, len(ids)),
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

    uvicorn.run(make_app(load_backend(cfg)), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
