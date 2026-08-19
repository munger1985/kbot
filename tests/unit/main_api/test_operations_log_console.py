"""开发日志目录、解析、分页和脱敏测试。"""

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import os
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from loguru import logger

from main_api.api.development_logs import router as development_logs_router
from main_api.app import _repository_path
from main_api.log_reader import LocalLogSearchService, LogQueryError
from platform_core.logger import LogConfig, LogManager


class OperationsLogConsoleTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.log_root = self.root / "logs"
        self.topology_path = self.root / "topology.toml"
        self.topology_path.write_text(
            """
[[processes]]
service_config = "main_api"
service_name = "kbot-main-api"

[[processes]]
service_config = "knowledge_core"
service_name = "kbot-knowledge-core-api"

[[processes]]
service_config = "aiops_agent"
service_name = "kbot-aiops-api"
""".strip(),
            encoding="utf-8",
        )
        service_dir = self.log_root / "main_api"
        service_dir.mkdir(parents=True)
        stamp = datetime.now(timezone.utc).astimezone().strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )[:-3]
        self.stamp = stamp
        (service_dir / "runtime.log").write_text(
            "\n".join(
                [
                    f"{stamp} | DEBUG    | [api] main_api.jobs:claim:10 - 准备领取任务",
                    f"{stamp} | INFO     | [api] main_api.jobs:claim:11 - 已领取任务 request_id=req-1",
                    f"{stamp} | INFO     | [worker] main_api.jobs:run:12 - 检索完成 | event=kc.discovery.completed | run_id=run-1 | task_id=task-1 | trace_id=trace-1 | job_id=job-1",
                    f"{stamp} | ERROR    | [worker] main_api.jobs:run:20 - 执行失败 error_id=error-1 password=bad-secret",
                    "Traceback (most recent call last):",
                    "  RuntimeError: boom Authorization=Bearer dangerous-token",
                ]
            ),
            encoding="utf-8",
        )
        (service_dir / "access.log").write_text(
            f"{stamp} | ERROR    | [api] API 访问 | method=POST | "
            "path=/api/v1/test | status=503 | duration_ms=12.50 | "
            "client=127.0.0.1 | request_id=req-2",
            encoding="utf-8",
        )
        aiops_dir = self.log_root / "aiops_agent"
        aiops_dir.mkdir(parents=True)
        (aiops_dir / "runtime.log").write_text(
            f"{stamp} | INFO     | [api] aiops_agent.bootstrap:ready:10 - "
            "AIOps API 已就绪",
            encoding="utf-8",
        )
        self.service = self._service()

    def _service(self, **overrides):
        return LocalLogSearchService(
            log_root=self.log_root,
            topology_path=self.topology_path,
            **overrides,
        )

    def _search(self, **filters):
        events, cursor, total = self.service.search(**filters)
        return events, cursor, total

    def tearDown(self):
        logger.remove()
        self.temporary_directory.cleanup()

    def test_relative_paths_use_deployment_resource_root(self):
        resource_dir = self.root / "resources"
        config_file = self.root / "configuration" / "kbot.toml"
        with patch.dict(
            os.environ,
            {
                "KBOT_RESOURCE_DIR": str(resource_dir),
                "KBOT_CONFIG_FILE": str(config_file),
            },
        ):
            self.assertEqual(
                self.root / "logs",
                _repository_path("./logs"),
            )
            self.assertEqual(
                resource_dir / "topology.toml",
                _repository_path("resources/topology.toml"),
            )

    def test_absolute_log_path_is_unchanged(self):
        absolute = self.root / "service-logs"
        self.assertEqual(absolute, _repository_path(str(absolute)))

    def test_catalog_groups_runtime_and_access_by_service(self):
        services = self.service.services()
        by_name = {item["service_name"]: item for item in services}
        self.assertEqual({"main_api", "aiops_agent"}, set(by_name))
        self.assertEqual("RUNTIME", by_name["main_api"]["runtime"]["stream"])
        self.assertEqual("ACCESS", by_name["main_api"]["access"]["stream"])
        self.assertEqual(
            "RUNTIME", by_name["aiops_agent"]["runtime"]["stream"]
        )

    def test_list_omits_raw_and_detail_is_redacted(self):
        events, _, _ = self._search(
            service_name="main_api", streams={"RUNTIME"},
        )
        error = next(event for event in events if event["level"] == "ERROR")
        self.assertNotIn("raw", error)
        detail = self.service.event_detail(event_id=error["event_id"])
        self.assertTrue(detail["has_traceback"])
        self.assertIn("RuntimeError: boom", detail["raw"])
        self.assertNotIn("bad-secret", detail["raw"])
        self.assertNotIn("dangerous-token", detail["raw"])

    def test_access_search_extracts_request_metrics(self):
        events, _, total = self._search(
            service_name="main_api", streams={"ACCESS"}, http_status=503,
        )
        self.assertEqual(1, total)
        self.assertEqual("api", events[0]["process"])
        self.assertEqual(12.5, events[0]["duration_ms"])
        self.assertEqual("req-2", events[0]["request_id"])

    def test_http_list_and_detail_contract(self):
        app = FastAPI()
        app.state.development_log_search_service = self.service
        app.include_router(development_logs_router)
        client = TestClient(app)
        listing = client.get(
            "/api/v1/development/logs/events",
            params={"service_name": "main_api", "stream": "RUNTIME"},
        )
        self.assertEqual(200, listing.status_code)
        payload = listing.json()
        self.assertEqual(4, payload["total"])
        self.assertNotIn("raw", payload["events"][0])
        error = next(
            row for row in payload["events"] if row["level"] == "ERROR"
        )
        detail = client.get(
            f"/api/v1/development/logs/events/{error['event_id']}"
        )
        self.assertEqual(200, detail.status_code)
        self.assertIn("raw", detail.json())
        self.assertNotIn("_search_text", detail.json())

    def test_filters_and_empty_level_selection(self):
        events, _, _ = self._search(
            service_name="main_api", streams={"RUNTIME"},
            levels={"INFO"}, filter_by_level=True,
            run_id="run-1", job_id="job-1", keyword="检索完成",
        )
        self.assertEqual(1, len(events))
        empty, _, _ = self._search(
            service_name="main_api", streams={"RUNTIME"},
            levels=set(), filter_by_level=True,
        )
        self.assertEqual([], empty)
        with self.assertRaises(LogQueryError):
            self._search(levels={"NOTICE"}, filter_by_level=True)

    def test_cursor_is_stable_and_bound_to_filters(self):
        first, cursor, total = self._search(
            service_name="main_api", streams={"RUNTIME"}, limit=2,
        )
        self.assertEqual(4, total)
        self.assertIsNotNone(cursor)
        second, _, second_total = self._search(
            service_name="main_api", streams={"RUNTIME"}, limit=2,
            cursor=cursor,
        )
        self.assertEqual(4, second_total)
        self.assertFalse(
            {row["event_id"] for row in first}
            & {row["event_id"] for row in second}
        )
        with self.assertRaises(LogQueryError):
            self._search(
                service_name="main_api", streams={"ACCESS"}, limit=2,
                cursor=cursor,
            )

    def test_export_is_bounded_and_never_contains_raw(self):
        service = self._service(max_export_events=2)
        rows = service.export(service_name="main_api", limit=999)
        self.assertEqual(2, len(rows))
        self.assertTrue(all("raw" not in row for row in rows))

    def test_json_redaction_is_recursive_and_malformed_line_is_safe(self):
        log_path = self.log_root / "main_api" / "runtime.log"
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": "INFO",
            "message": "结构化事件",
            "extra": {
                "authorization": "Bearer token",
                "nested": {
                    "database_username": "system",
                    "dsn": "oracle://admin:pw@db/service",
                    "prompt": "x" * 10000,
                },
            },
        }
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write("\n" + json.dumps(record, ensure_ascii=False))
            stream.write('\n{"malformed": true, "password": "visible"')
        events, _, _ = self._search(service_name="main_api")
        structured = next(row for row in events if row["message"] == "结构化事件")
        detail = self.service.event_detail(event_id=structured["event_id"])
        serialized = json.dumps(detail, ensure_ascii=False)
        self.assertNotIn("system", serialized)
        self.assertNotIn("admin:pw", serialized)
        self.assertNotIn("x" * 100, serialized)
        self.assertIn("[REDACTED]", serialized)
        malformed = next(row for row in events if '"malformed"' in row["message"])
        malformed_detail = self.service.event_detail(event_id=malformed["event_id"])
        self.assertNotIn("visible", malformed_detail["raw"])

    def test_scan_is_tail_bounded_and_rotation_event_id_is_stable(self):
        log_path = self.log_root / "main_api" / "runtime.log"
        log_path.write_text(
            f"{self.stamp} | INFO     | [api] old:1 - SHOULD_NOT_BE_SCANNED\n"
            + ("padding\n" * 400)
            + f"{self.stamp} | INFO     | [api] recent:2 - recent-event\n",
            encoding="utf-8",
        )
        service = self._service(max_bytes_per_file=512, max_total_scan_bytes=512)
        events, _, _ = service.search(service_name="main_api")
        self.assertFalse(any("SHOULD_NOT_BE_SCANNED" in row["message"] for row in events))
        recent = next(row for row in events if row["message"] == "recent-event")
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"{self.stamp} | INFO     | [api] later:3 - later-event\n")
        events, _, _ = service.search(service_name="main_api")
        self.assertEqual(
            recent["event_id"],
            next(row["event_id"] for row in events if row["message"] == "recent-event"),
        )

    def test_uncontrolled_directory_and_missing_root_are_ignored(self):
        rogue = self.log_root / "foreign_admin"
        rogue.mkdir()
        (rogue / "runtime.log").write_text("secret", encoding="utf-8")
        self.assertNotIn(
            "foreign_admin",
            {item["service_name"] for item in self.service.services()},
        )
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "runtime.log").write_text("secret", encoding="utf-8")
        linked = self.log_root / "knowledge_core"
        linked.symlink_to(outside, target_is_directory=True)
        self.assertNotIn(
            "knowledge_core",
            {item["service_name"] for item in self.service.services()},
        )
        missing = LocalLogSearchService(
            log_root=self.root / "missing",
            topology_path=self.topology_path,
        )
        self.assertEqual([], missing.services())
        self.assertEqual(([], None, 0), missing.search())

    def test_correlated_search_extracts_run_task_and_event(self):
        events = self.service.search_correlated(identifiers={"run-1"})
        self.assertEqual(1, len(events))
        self.assertEqual("task-1", events[0]["task_id"])
        self.assertEqual("kc.discovery.completed", events[0]["event_name"])

    def test_log_manager_separates_runtime_and_access(self):
        LogManager(
            LogConfig(
                service="knowledge_core",
                process="parser",
                log_dir=str(self.log_root),
                level="DEBUG",
                rotation="1 MB",
                retention="1 day",
                console_output=False,
            )
        ).setup()
        logger.info("解析器已启动")
        logger.bind(log_type="access").info(
            "API 访问 | status=200 | duration_ms=1.2"
        )
        runtime = (self.log_root / "knowledge_core" / "runtime.log").read_text(
            encoding="utf-8"
        )
        access = (self.log_root / "knowledge_core" / "access.log").read_text(
            encoding="utf-8"
        )
        self.assertIn("解析器已启动", runtime)
        self.assertNotIn("API 访问", runtime)
        self.assertIn("API 访问", access)


if __name__ == "__main__":
    unittest.main()
