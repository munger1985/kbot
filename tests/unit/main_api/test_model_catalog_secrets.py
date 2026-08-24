"""公开模型目录的嵌套凭据脱敏测试。"""

import unittest
from types import SimpleNamespace

from main_api.api.models import load_model_catalog


class _Client:
    async def list_models(self):
        return [{
            "model_id": "01900000-0000-7000-8000-000000000001",
            "served_model_name": "oci-model",
            "display_name": "OCI 模型",
            "category": 1,
            "provider": "oci",
            "status": "ACTIVE",
            "model_params": {
                "max_tokens": 8192,
                "config_file": {"key_content": "private-key"},
            },
        }]


class ModelCatalogSecretsTest(unittest.IsolatedAsyncioTestCase):
    async def test_nested_oci_credentials_are_not_exposed(self):
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(model_config_clients=(_Client(),))
            )
        )

        rows = await load_model_catalog(request)

        self.assertEqual(8192, rows[0]["model_params"]["max_tokens"])
        self.assertNotIn("config_file", rows[0]["model_params"])
        self.assertNotIn("private-key", repr(rows))


if __name__ == "__main__":
    unittest.main()
