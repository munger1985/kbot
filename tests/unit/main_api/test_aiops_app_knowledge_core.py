"""AIOps Knowledge Core BFF 的模型配置测试。"""

from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import UUID

from fastapi import HTTPException

from main_api.api.aiops_app import (
    AIOpsCollectionModelsPayload,
    _fixed_manual_collection,
    _validated_aiops_models,
    update_aiops_knowledge_core_models,
)
from platform_core.contracts import AuthContext, PrincipalKind


COLLECTION_ID = UUID("019ffff0-0000-7000-8000-000000000001")
EMBEDDING_ID = UUID("019ffff0-0000-7000-8000-000000000002")
VLM_ID = UUID("019ffff0-0000-7000-8000-000000000003")


class _KnowledgeClient:
    def __init__(self, collections=None):
        self.collections = collections or []
        self.updated = None

    async def list_collections(self, **_):
        return {"collections": self.collections}

    async def update_collection_models(self, **kwargs):
        self.updated = kwargs
        return {"collection_id": str(COLLECTION_ID), "row_version": 8}


def _request(client):
    context = AuthContext(
        principal_kind=PrincipalKind.PORTAL,
        client_id="aiops-user-session",
        api_key_id="aiops-user-token",
        request_id="aiops-kc-test",
        trace_id="aiops-kc-test",
        app_id="aiops",
        domain_id="42",
        asserted_user_id="aiopsadmin",
    )
    return SimpleNamespace(
        state=SimpleNamespace(auth_context=context),
        app=SimpleNamespace(
            state=SimpleNamespace(knowledge_core_client=client)
        ),
    )


class AIOpsKnowledgeCoreBffTest(unittest.IsolatedAsyncioTestCase):
    async def test_fixed_collection_requires_fixed_aiops_metadata(self):
        client = _KnowledgeClient([
            {
                "collection_id": str(COLLECTION_ID),
                "display_name": "operations-manuals",
                "status": "ACTIVE",
                "metadata": {
                    "owner_app_id": "aiops",
                    "fixed_resource": True,
                },
            }
        ])

        domain_id, collection = await _fixed_manual_collection(
            _request(client), require_active=True
        )

        self.assertEqual(42, domain_id)
        self.assertEqual(str(COLLECTION_ID), collection["collection_id"])

    async def test_model_validation_rejects_wrong_category(self):
        with patch(
            "main_api.api.aiops_app.load_model_catalog",
            AsyncMock(return_value=[{
                "model_id": str(EMBEDDING_ID),
                "category": 1,
            }]),
        ):
            with self.assertRaises(HTTPException) as raised:
                await _validated_aiops_models(
                    object(),
                    parser_vlm=None,
                    embedding=EMBEDDING_ID,
                    visual_embedding=None,
                )

        self.assertEqual(422, raised.exception.status_code)

    async def test_model_update_forwards_validated_roles_and_row_version(self):
        client = _KnowledgeClient()
        request = _request(client)
        payload = AIOpsCollectionModelsPayload(
            parser_vlm=VLM_ID,
            embedding=EMBEDDING_ID,
            visual_embedding=None,
            expected_row_version=7,
        )
        collection = {
            "collection_id": str(COLLECTION_ID),
            "display_name": "operations-manuals",
        }
        validated = {
            "parser_vlm": str(VLM_ID),
            "embedding": str(EMBEDDING_ID),
        }
        with (
            patch(
                "main_api.api.aiops_app._require",
                AsyncMock(return_value=(42, "aiopsadmin", object())),
            ),
            patch(
                "main_api.api.aiops_app._fixed_manual_collection",
                AsyncMock(return_value=(42, collection)),
            ),
            patch(
                "main_api.api.aiops_app._validated_aiops_models",
                AsyncMock(return_value=validated),
            ),
        ):
            response = await update_aiops_knowledge_core_models(
                payload, request
            )

        self.assertEqual(8, response["row_version"])
        self.assertEqual(
            {"models": validated, "expected_row_version": 7},
            client.updated["payload"],
        )
        self.assertEqual(COLLECTION_ID, client.updated["collection_id"])


if __name__ == "__main__":
    unittest.main()
