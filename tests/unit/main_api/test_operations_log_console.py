"""开发日志目录、解析和筛选测试。"""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from loguru import logger

from main_api.developer_tools import LocalLogSearchService
from platform_core.logger import LogConfig, LogManager


class OperationsLogConsoleTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        self.log_root = Path(self.temporary_directory.name)
        service_dir = self.log_root / "main_api"
        service_dir.mkdir()
        (service_dir / "runtime.log").write_text(
            "\n".join(
                [
                    "2026-07-27 10:00:00.001 | DEBUG    | [api] main_api.jobs:claim:10 - 准备领取任务",
                    "2026-07-27 10:00:00.002 | INFO     | [api] main_api.jobs:claim:11 - 已领取任务 request_id=req-1",
                    "2026-07-27 10:00:00.003 | ERROR    | [worker] main_api.jobs:run:20 - 执行失败 error_id=019f8eae-2c25-7d48-b044-350ec3f5a001",
                    "Traceback (most recent call last):",
                    "  RuntimeError: boom",
                ]
            ),
            encoding="utf-8",
        )
        (service_dir / "access.log").write_text(
            "2026-07-27 10:00:01.001 | ERROR    | [api] "
            "API 访问 | method=POST | path=/api/v1/test | status=503 | "
            "duration_ms=12.50 | client=127.0.0.1 | request_id=req-2",
            encoding="utf-8",
        )
        self.service = LocalLogSearchService(log_root=self.log_root)

    def tearDown(self):
        logger.remove()
        self.temporary_directory.cleanup()

    def test_catalog_groups_runtime_and_access_by_service(self):
        services = self.service.services()
        self.assertEqual(1, len(services))
        self.assertEqual("main_api", services[0]["service_name"])
        self.assertEqual("RUNTIME", services[0]["runtime"]["log_type"])
        self.assertEqual("ACCESS", services[0]["access"]["log_type"])

    def test_runtime_search_keeps_process_and_multiline_error(self):
        events = self.service.search(
            service_name="main_api",
            log_type="RUNTIME",
        )
        self.assertEqual(
            {"DEBUG", "INFO", "ERROR"},
            {event["level"] for event in events},
        )
        error = next(event for event in events if event["level"] == "ERROR")
        self.assertEqual("worker", error["process"])
        self.assertTrue(error["has_traceback"])
        self.assertIn("RuntimeError: boom", error["raw"])
        self.assertEqual(
            "019f8eae-2c25-7d48-b044-350ec3f5a001",
            error["error_id"],
        )

    def test_access_search_extracts_request_metrics(self):
        events = self.service.search(
            service_name="main_api",
            log_type="ACCESS",
        )
        self.assertEqual(1, len(events))
        self.assertEqual("api", events[0]["process"])
        self.assertEqual(503, events[0]["http_status"])
        self.assertEqual(12.5, events[0]["duration_ms"])
        self.assertEqual("req-2", events[0]["request_id"])

    def test_search_filters_level_and_keyword(self):
        events = self.service.search(
            service_name="main_api",
            log_type="RUNTIME",
            levels={"INFO"},
            keyword="req-1",
        )
        self.assertEqual(1, len(events))
        self.assertEqual("req-1", events[0]["request_id"])

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

        runtime = (
            self.log_root / "knowledge_core" / "runtime.log"
        ).read_text(encoding="utf-8")
        access = (
            self.log_root / "knowledge_core" / "access.log"
        ).read_text(encoding="utf-8")
        self.assertIn("[parser]", runtime)
        self.assertIn("解析器已启动", runtime)
        self.assertNotIn("API 访问", runtime)
        self.assertIn("[parser]", access)
        self.assertIn("API 访问", access)


if __name__ == "__main__":
    unittest.main()
