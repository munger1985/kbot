import unittest
from unittest.mock import patch

from platform_core.database.oracle import create_database_runtime


class PlatformCoreRuntimeTest(unittest.IsolatedAsyncioTestCase):
    async def test_app_owned_runtime_can_be_created_and_closed(self):
        with patch.dict(
            "os.environ", {"KBOT_ORACLE_PASSWORD": "test-password"}
        ):
            runtime = create_database_runtime()
        try:
            self.assertIsNotNone(runtime.session_factory)
            self.assertIsNotNone(runtime.engine)
        finally:
            await runtime.close()


if __name__ == "__main__":
    unittest.main()
