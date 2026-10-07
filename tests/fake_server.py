"""Stdlib-only fake of the server API the harness talks to (vLLM / HF L1).

Streams `tokens` content chunks per request with a small delay, exposes /health,
/metrics (Prometheus text) and /reset_prefix_cache.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from server import openai_format as fmt


class FakeState:
    def __init__(self, delay_s: float = 0.002, fail_every: int = 0) -> None:
        self.delay_s = delay_s
        self.fail_every = fail_every
        self.requests = 0
        self.resets = 0
        self.bodies: list[dict] = []
        self.lock = threading.Lock()


def make_handler(state: FakeState):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # silence test output
            pass

        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, b"{}")
            elif self.path == "/metrics":
                text = (
                    "# HELP vllm:prompt_tokens_total x\n"
                    f'vllm:prompt_tokens_total{{model_name="m"}} {state.requests * 10}\n'
                    f'vllm:num_preemptions_total{{model_name="m"}} 0\n'
                    'vllm:kv_cache_usage_perc{model_name="m"} 0.42\n'
                    'vllm:num_requests_running{model_name="m"} 3\n'
                )
                self._send(200, text.encode(), "text/plain")
            else:
                self._send(404, b"{}")

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/reset_prefix_cache":
                with state.lock:
                    state.resets += 1
                self._send(200, b"{}")
                return
            if self.path != "/v1/chat/completions":
                self._send(404, b"{}")
                return
            with state.lock:
                state.requests += 1
                n = state.requests
                state.bodies.append(body)
            if state.fail_every and n % state.fail_every == 0:
                self._send(500, b'{"error":"boom"}')
                return
            max_tokens = int(body.get("max_tokens", 8))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            cid = fmt.new_completion_id()
            self.wfile.write(fmt.chunk(cid, "m", role="assistant", content="").encode())
            # Echo the last user message's id-ish content so outputs are deterministic.
            for i in range(max_tokens):
                time.sleep(state.delay_s)
                self.wfile.write(fmt.chunk(cid, "m", content=f"t{i} ").encode())
                self.wfile.flush()
            self.wfile.write(fmt.chunk(cid, "m", finish_reason="length").encode())
            u = fmt.usage(17, max_tokens)
            self.wfile.write(fmt.chunk(cid, "m", include_choice=False, usage_stats=u).encode())
            self.wfile.write(fmt.DONE.encode())
            self.wfile.flush()
            self.close_connection = True

    return Handler


class FakeServer:
    def __init__(self, **kwargs) -> None:
        self.state = FakeState(**kwargs)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.state))
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> FakeServer:
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
