from __future__ import annotations

import hashlib
import hmac
import os
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


MAX_BODY_BYTES = 2 * 1024 * 1024


def _read_secret(name: str) -> str:
    value = Path(f"/run/secrets/{name}").read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"Secret为空：{name}")
    return value


WEBHOOK_KEY = _read_secret("kbot_webhook_key")
WEBHOOK_SECRET = _read_secret("kbot_webhook_secret")
BASE_URL = os.environ["KBOT_WEBHOOK_BASE_URL"].rstrip("/")
DESTINATION = f"{BASE_URL}/api/v1/integrations/aiops/signals/{WEBHOOK_KEY}"


class Handler(BaseHTTPRequestHandler):
    server_version = "KBotWebhookSigner/1.0"

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/healthz":
            self.send_error(404)
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok\n")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/alertmanager":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(400, "Content-Length无效")
            return
        if length <= 0 or length > MAX_BODY_BYTES:
            self.send_error(413, "Webhook正文大小无效")
            return
        body = self.rfile.read(length)
        timestamp = str(int(time.time()))
        signature = hmac.new(
            WEBHOOK_SECRET.encode("utf-8"),
            timestamp.encode("ascii") + b"." + body,
            hashlib.sha256,
        ).hexdigest()
        request = urllib.request.Request(
            DESTINATION,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-KBot-Timestamp": timestamp,
                "X-KBot-Signature": f"sha256={signature}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                payload = response.read()
                self.send_response(response.status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)
        except urllib.error.HTTPError as exc:
            self.send_response(exc.code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(exc.read())
        except (OSError, urllib.error.URLError) as exc:
            # 不输出目标URL、Webhook Key或Secret，只报告异常类型。
            print(f"KBot Webhook转发失败：{type(exc).__name__}", flush=True)
            self.send_error(502, "KBot事件入口暂时不可用")

    def log_message(self, format: str, *args: object) -> None:
        # 避免默认访问日志记录包含Webhook路由或其他敏感信息。
        return


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
