"""4.0 可部署进程拓扑测试。"""

import unittest
from pathlib import Path

from main_api.entrypoints.notification_worker import PROCESS_NAME
from platform_core.logger import LogConfig
from tests.acceptance.check_process_topology import (
    check_process_topology,
    load_processes,
)


class ProcessTopologyTest(unittest.TestCase):
    def test_notification_worker_uses_valid_log_process_name(self):
        config = LogConfig(
            service="main_api",
            process=PROCESS_NAME,
        )

        self.assertEqual("notification_worker", config.process)

    def test_every_app_has_configured_process(self):
        self.assertEqual([], check_process_topology())
        processes = load_processes()

        self.assertEqual(22, len(processes))
        self.assertEqual(
            5,
            sum(item["kind"] == "worker" for item in processes),
        )

    def test_development_ui_is_managed_by_start_and_stop_scripts(self):
        root = Path(__file__).resolve().parents[2]
        start = (root / "start_kbot.sh").read_text(encoding="utf-8")
        stop = (root / "stop_kbot.sh").read_text(encoding="utf-8")

        self.assertIn(
            'KBOT_CONFIG_FILE="${KBOT_CONFIG_FILE:-configuration/kbot.toml}"',
            start,
        )
        self.assertNotIn("export PYTHONPATH=", start)
        self.assertIn("editable package", start)
        self.assertIn('ENVIRONMENT="${ENVIRONMENT:-$CONFIG_ENVIRONMENT}"', start)
        self.assertIn("python scripts/deployment/check_deployment.py", start)
        self.assertIn('KBOT_UI_ENABLED="${KBOT_UI_ENABLED:-true}"', start)
        self.assertIn(
            'python3 tools/dev_console/server.py --port "$port"',
            start,
        )
        ui_server = (
            root / "tools" / "dev_console" / "server.py"
        ).read_text(encoding="utf-8")
        self.assertIn('request_path.startswith("/ui/")', ui_server)
        self.assertIn('"/ui/km/login.html"', ui_server)
        for group in (
            "Model Serving",
            "Knowledge Core",
            "Knowledge Retrieval App",
            "KM Asset App",
            "Agent Runtime",
            "AIOps Agent",
            "Main API",
            "Development Tools",
        ):
            self.assertIn(group, start)
        self.assertIn(
            'IFS=\':\' read -r group name log_service module <<< "$service"',
            start,
        )
        self.assertIn('local log_file="${log_dir}/runtime.log"', start)
        self.assertIn(
            '"km_asset_app.entrypoints.worker"',
            start,
        )
        self.assertIn("无监听端口", start)
        self.assertNotIn("logs/startup", start)
        self.assertIn("get_ui_pid()", stop)
        self.assertIn("tools/dev_console/server\\\\.py", stop)
        self.assertIn("http\\\\.server", stop)


if __name__ == "__main__":
    unittest.main()
