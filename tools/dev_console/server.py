"""同时发布既有开发测试台与仓库 ui/ 正式页面。"""

from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[2]
DEV_ROOT = ROOT / "tools" / "dev_console"
UI_ROOT = ROOT / "ui"


class KBotUiHandler(SimpleHTTPRequestHandler):
    """保持测试台根路径不变，并把 /ui/ 映射到正式 UI 目录。"""

    def _redirect_ui_root(self) -> bool:
        path = urlsplit(self.path).path
        if path in {"/ui", "/ui/", "/km", "/km/"}:
            self.send_response(302)
            self.send_header("Location", "/ui/km/dashboard.html")
            self.end_headers()
            return True
        return False

    def do_GET(self):
        if self._redirect_ui_root():
            return
        super().do_GET()

    def do_HEAD(self):
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--bind", default="0.0.0.0")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.bind, args.port), KBotUiHandler)
    print(f"KBot UI 服务已启动：http://{args.bind}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
