"""锁定 KBot 4.1 不再包含 KM 产品线。"""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class KmRetirementTest(unittest.TestCase):
    def test_km_runtime_and_ui_are_absent(self) -> None:
        for relative in (
            "services/km_asset_app",
            "services/agent_runtime/src/agent_runtime/specialists/km_asset",
            "services/main_api/src/main_api/api/km_asset_app.py",
            "database/oracle/km_asset_app",
            "database/oracle/bootstrap/km_asset",
            "ui/km",
        ):
            self.assertFalse((ROOT / relative).exists(), relative)

    def test_workspace_and_topology_do_not_register_km(self) -> None:
        for relative in (
            "pyproject.toml",
            "resources/topology.toml",
            "installation/catalog.toml",
            "configuration/oracle_schema_services.ini",
        ):
            source = (ROOT / relative).read_text(encoding="utf-8").lower()
            self.assertNotIn("km_asset", source, relative)
            self.assertNotIn("km-asset", source, relative)

    def test_openapi_does_not_expose_km_routes(self) -> None:
        for path in (ROOT / "docs" / "openapi").glob("*.json"):
            source = path.read_text(encoding="utf-8").lower()
            self.assertNotIn("/km-asset", source, path.name)
            self.assertNotIn("km-assets", source, path.name)


if __name__ == "__main__":
    unittest.main()
