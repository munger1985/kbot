"""KBot 4.0 开发日志页面的静态契约检查。"""

from html.parser import HTMLParser
from http.client import HTTPConnection
from pathlib import Path
import subprocess
from threading import Thread
import unittest

from tools.dev_console.server import KBotUiHandler, KBotUiServer


ROOT = Path(__file__).resolve().parents[2]
UI_ROOT = ROOT / "tools" / "dev_console"


class _PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids: set[str] = set()
        self.scripts: list[str] = []
        self.links: list[str] = []
        self.app_links: dict[str, str] = {}

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(values["id"])
        if tag == "script" and values.get("src"):
            self.scripts.append(values["src"])
        if tag == "link" and values.get("href"):
            self.links.append(values["href"])
        if tag == "a" and values.get("data-app") and values.get("href"):
            self.app_links[values["data-app"]] = values["href"]


class UiStaticPagesTest(unittest.TestCase):
    def test_home_serves_the_app_directory_without_redirecting(self):
        expected = {
            "knowledge_retrieval": "/ui/knowledge_retrieval/login.html",
            "aiops": "/ui/aiops/login.html",
            "media_studio": "/ui/media_studio/login.html",
        }
        with KBotUiServer(("127.0.0.1", 0), KBotUiHandler) as server:
            worker = Thread(
                target=server.serve_forever, kwargs={"poll_interval": 0.05}
            )
            worker.start()
            connection = HTTPConnection(*server.server_address, timeout=3)
            try:
                for path in ("/", "/index.html", "/ui/", "/ui/index.html"):
                    connection.request("GET", path)
                    response = connection.getresponse()
                    self.assertEqual(200, response.status, path)
                    self.assertIsNone(response.getheader("Location"), path)
                    parser = _PageParser()
                    parser.feed(response.read().decode("utf-8"))
                    self.assertEqual(expected, parser.app_links)
                for path in (*expected.values(), "/operations-logs.html"):
                    connection.request("HEAD", path)
                    self.assertEqual(200, connection.getresponse().status, path)
            finally:
                connection.close()
                server.shutdown()
                worker.join()

    def test_only_log_viewer_page_remains(self):
        self.assertEqual(
            ["operations-logs.html"],
            sorted(path.name for path in UI_ROOT.glob("*.html")),
        )
        self.assertEqual(
            ["operations-logs.js", "shared.js"],
            sorted(path.name for path in UI_ROOT.glob("*.js")),
        )

    def test_log_page_references_existing_local_assets(self):
        page = UI_ROOT / "operations-logs.html"
        parser = _PageParser()
        parser.feed(page.read_text(encoding="utf-8"))
        for reference in [*parser.scripts, *parser.links]:
            if reference.startswith("./"):
                asset_path = reference[2:].split("?", 1)[0]
                self.assertTrue(
                    (UI_ROOT / asset_path).is_file(),
                    f"日志页面缺少资源 {reference}",
                )

    def test_log_page_exposes_required_controls(self):
        parser = _PageParser()
        parser.feed(
            (UI_ROOT / "operations-logs.html").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(
            {
                "log-filter-form",
                "service-filter",
                "log-type-filter",
                "refresh-interval",
                "log-rows",
                "event-detail",
            }.issubset(parser.ids)
        )

    def test_javascript_syntax(self):
        for script in ("shared.js", "operations-logs.js"):
            result = subprocess.run(
                ["node", "--check", str(UI_ROOT / script)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(0, result.returncode, result.stderr)

    def test_log_page_has_no_connection_or_identity_form(self):
        html = (UI_ROOT / "operations-logs.html").read_text(
            encoding="utf-8"
        )
        script = (UI_ROOT / "operations-logs.js").read_text(
            encoding="utf-8"
        )
        shared = (UI_ROOT / "shared.js").read_text(encoding="utf-8")
        for value in (
            'id="auth-form"',
            "Main API URL",
            "Domain ID",
            "User ID",
        ):
            self.assertNotIn(value, html)
        self.assertNotIn("bindAuthForm", script)
        self.assertNotIn("localStorage", shared)
        self.assertNotIn("X-KBot-Domain-ID", shared)
        self.assertNotIn("X-KBot-Test-Auth", shared)
        self.assertNotIn("X-KBot-User-ID", shared)
        self.assertNotIn("Authorization:", shared)
        self.assertNotIn("apiKey", shared)

    def test_development_log_page_includes_two_logs_and_all_levels(self):
        html = (UI_ROOT / "operations-logs.html").read_text(
            encoding="utf-8"
        )
        script = (UI_ROOT / "operations-logs.js").read_text(
            encoding="utf-8"
        )
        for value in (
            "RUNTIME",
            "ACCESS",
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        ):
            self.assertIn(f'value="{value}"', html)
        self.assertIn("/api/v1/development/logs/events", script)
        self.assertIn(
            "/api/v1/development/logs/events/${encodeURIComponent(eventId)}",
            script,
        )
        self.assertIn("service_name: event.service_name", script)
        self.assertIn("stream: event.stream", script)
        self.assertIn("service.runtime?.source_file", script)
        self.assertIn("正在读取完整日志", script)
        self.assertIn("/api/v1/development/logs/services", script)
        self.assertIn("KBotUI.developmentLogApi", script)
        self.assertIn('params.set("stream", logTypeFilter.value)', script)
        self.assertNotIn(
            'params.set("log_type", logTypeFilter.value)', script
        )

if __name__ == "__main__":
    unittest.main()
