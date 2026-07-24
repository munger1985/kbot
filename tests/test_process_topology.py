"""4.0 可部署进程拓扑测试。"""

import unittest

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


if __name__ == "__main__":
    unittest.main()
