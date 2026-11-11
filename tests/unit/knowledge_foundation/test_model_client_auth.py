"""原生表示客户端必须遵循模型服务认证，而不是纯计算服务的认证合同。"""

import unittest
from unittest.mock import patch

from platform_clients.knowledge_computation import KnowledgeComputationClient
from platform_clients.knowledge_models import NativeRepresentationClient


class ModelClientAuthTests(unittest.TestCase):
    def test_model_auth_is_regenerated_for_each_request_with_exact_audience_and_context(self):
        context = object()
        client = NativeRepresentationClient(base_url="http://models.invalid", caller_service="aiops-api",
            audience="model-embedding", service_scopes=("model.inference",))
        with patch("platform_clients.knowledge_models.build_internal_auth_headers",
                   side_effect=[{"auth":"first"}, {"auth":"second"}]) as model_auth, patch(
                "platform_clients.knowledge_computation.build_scoped_internal_auth_headers") as compute_auth:
            self.assertEqual(client._auth_headers(context), {"auth":"first"})
            self.assertEqual(client._auth_headers(context), {"auth":"second"})
            self.assertEqual(model_auth.call_count, 2)
            model_auth.assert_called_with(audience="model-embedding", caller_service="aiops-api", context=context)
            compute_auth.assert_not_called()

    def test_compute_auth_keeps_short_lived_scoped_service_identity(self):
        context = object()
        client = KnowledgeComputationClient(base_url="http://compute.invalid", caller_service="aiops-api",
            audience="knowledge-core")
        with patch("platform_clients.knowledge_computation.build_scoped_internal_auth_headers",
                   return_value={"auth":"scoped"}) as compute_auth:
            self.assertEqual(client._auth_headers(context), {"auth":"scoped"})
            compute_auth.assert_called_once_with(audience="knowledge-core", caller_service="aiops-api",
                scopes=("knowledge.parse.compute",), context=context)


if __name__ == "__main__":
    unittest.main()
