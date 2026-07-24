"""4.0 可部署进程拓扑测试。"""

import unittest
from pathlib import Path

from scripts.check_process_topology import (
    check_process_topology,
    load_processes,
)


class ProcessTopologyTest(unittest.TestCase):
    def test_every_app_has_configured_process(self):
        self.assertEqual([], check_process_topology())
        processes = load_processes()

        self.assertEqual(14, len(processes))
        self.assertEqual(
            2,
            sum(item["kind"] == "worker" for item in processes),
        )

    def test_development_ui_is_managed_by_start_and_stop_scripts(self):
        root = Path(__file__).resolve().parents[1]
        start = (root / "start_kbot.sh").read_text(encoding="utf-8")
        stop = (root / "stop_kbot.sh").read_text(encoding="utf-8")

        self.assertIn('ENVIRONMENT="${ENVIRONMENT:-development}"', start)
        self.assertIn('KBOT_UI_ENABLED="${KBOT_UI_ENABLED:-true}"', start)
        self.assertIn('python3 -m http.server "$port" -d ui', start)
        for group in (
            "Model Serving",
            "Knowledge Core",
            "Agent Runtime",
            "AIOps Agent",
            "Main API",
            "Development Tools",
        ):
            self.assertIn(group, start)
        self.assertIn(
            'IFS=\':\' read -r group name script dir <<< "$service"',
            start,
        )
        self.assertIn("get_ui_pid()", stop)
        self.assertIn("http\\\\.server", stop)


if __name__ == "__main__":
    unittest.main()
