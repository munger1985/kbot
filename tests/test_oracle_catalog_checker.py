"""Oracle 共享 Schema 对象清单检查器测试。"""

import unittest
from unittest.mock import patch

from scripts.check_oracle_catalog import expected_objects
from scripts.check_oracle_schema import SERVICE_TABLES, SERVICE_VIEWS
from scripts.oracle_preflight import require_oracle_listener


class OracleCatalogCheckerTest(unittest.TestCase):
    def test_expected_catalog_covers_every_service_owner(self):
        objects = expected_objects()

        self.assertEqual(
            sum(len(value) for value in SERVICE_TABLES.values())
            + sum(len(value) for value in SERVICE_VIEWS.values()),
            len(objects),
        )
        self.assertEqual(objects["KBOT_AGENT_RUN"], "TABLE")
        self.assertEqual(objects["KBOT_V_OPS_RUN"], "VIEW")

    @patch(
        "scripts.oracle_preflight.socket.create_connection",
        side_effect=TimeoutError,
    )
    def test_listener_preflight_fails_without_creating_pool(
        self, create_connection
    ):
        with self.assertRaisesRegex(RuntimeError, "Listener 不可达"):
            require_oracle_listener(
                host="127.0.0.1",
                port=1521,
                timeout_seconds=0.1,
            )

        create_connection.assert_called_once_with(
            ("127.0.0.1", 1521), timeout=0.1
        )


if __name__ == "__main__":
    unittest.main()
