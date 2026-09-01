"""同时发布开发日志页面与仓库 ui/ KM 正式页面。"""

from __future__ import annotations

import argparse
import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

try:
    import tomllib as tomli
except ModuleNotFoundError:  # Python 3.10
    import tomli


ROOT = Path(__file__).resolve().parents[2]
DEV_ROOT = ROOT / "tools" / "dev_console"
UI_ROOT = ROOT / "ui"


def _load_main_api_base_url() -> str:
    """从统一部署配置读取浏览器使用的 Main API 地址。"""
    configured = Path(
        os.getenv("KBOT_CONFIG_FILE") or ROOT / "configuration" / "kbot.toml"
    )
    if not configured.is_absolute():
        configured = (ROOT / configured).resolve()
    if not configured.is_file():
        raise RuntimeError(f"未找到部署配置：{configured}")
    with configured.open("rb") as stream:
        deployment = tomli.load(stream)
    value = str((deployment.get("ui") or {}).get("main_api_base_url") or "")
    normalized = value.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError(
            "kbot.toml 必须配置合法的 [ui].main_api_base_url"
        )
    return normalized


class KBotUiHandler(SimpleHTTPRequestHandler):
    """发布日志页，并把 /ui/ 映射到正式 KM UI 目录。"""

    main_api_base_url = ""

    def end_headers(self) -> None:
        """开发 UI 始终重新校验静态资源，避免部署后继续执行旧认证脚本。"""
        path = urlsplit(self.path).path
        if path.startswith("/ui/") and path != "/ui/runtime-config.js":
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _empty_favicon(self) -> bool:
        if urlsplit(self.path).path != "/favicon.ico":
            return False
        self.send_response(204)
        self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()
        return True

    def _redirect_ui_root(self) -> bool:
        path = urlsplit(self.path).path
        if path in {"/ui", "/ui/", "/km", "/km/"}:
            self.send_response(302)
            self.send_header("Location", "/ui/km/login.html")
            self.end_headers()
            return True
        return False

    def _redirect_log_root(self) -> bool:
        """开发工具根路径只进入唯一保留的日志页面。"""
        if urlsplit(self.path).path not in {"", "/"}:
            return False
        self.send_response(302)
        self.send_header("Location", "/operations-logs.html")
        self.end_headers()
        return True

    def do_GET(self):
        if self._empty_favicon():
            return
        if self._redirect_log_root():
            return
        if urlsplit(self.path).path == "/ui/runtime-config.js":
            payload = (
                "globalThis.KBOT_UI_CONFIG = Object.freeze("
                + json.dumps(
                    {"mainApiBaseUrl": self.main_api_base_url},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + ");\n"
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/javascript; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self._redirect_ui_root():
            return
        super().do_GET()

    def do_HEAD(self):
        if self._empty_favicon():
            return
        if self._redirect_log_root():
            return
        if self._redirect_ui_root():
            return
        super().do_HEAD()

    def translate_path(self, path: str) -> str:
        request_path = unquote(urlsplit(path).path)
        if request_path.startswith("/ui/"):
            root = UI_ROOT
            relative = request_path.removeprefix("/ui/")
        else:
            root = DEV_ROOT
            relative = request_path.lstrip("/")
        candidate = (root / relative).resolve()
        if candidate != root and root not in candidate.parents:
            return str(root / "__forbidden__")
        return str(candidate)


class KBotUiServer(ThreadingHTTPServer):
    """限制慢连接生命周期，避免开发 UI 被长期占满。"""

    allow_reuse_address = True
    daemon_threads = True
    block_on_close = False
    request_queue_size = 128
    request_timeout_seconds = 15

    def get_request(self):
        request, client_address = super().get_request()
        request.settimeout(self.request_timeout_seconds)
        return request, client_address


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--bind", default="0.0.0.0")
    args = parser.parse_args()
    KBotUiHandler.main_api_base_url = _load_main_api_base_url()
    server = KBotUiServer((args.bind, args.port), KBotUiHandler)
    print(
        f"KBot UI 服务已启动：http://{args.bind}:{args.port}"
        f" | Main API={KBotUiHandler.main_api_base_url}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
