"""KM Asset 固定问数模型与执行边界测试。"""

import unittest
from types import SimpleNamespace
from uuid import UUID

from pydantic import ValidationError

from km_asset_app.application import KmAgentService, KmAssetApplicationError, KmAssetService
from km_asset_app.integrations import SharePointClient
from km_asset_app.api.assets import SourceUpdateRequest
from data_query.application.managed_datasets import km_asset_definition
from data_query.contracts import DataQueryPlanV1, SemanticModelDefinition
from data_query.connectors import compile_dialect_query
from platform_core.contracts import AgentExecutionSpec


class KmAssetCatalogTest(unittest.TestCase):
    def test_catalog_is_managed_and_domain_scoped(self):
        model = SemanticModelDefinition.model_validate(
            km_asset_definition(schema_name="KBOTUI_DEV")
        )
        self.assertEqual("KBOT_V_KM_ASSET_CURRENT", model.datasets[0].physical_object)
        self.assertEqual("DOMAIN_ID", model.datasets[0].scope_column)

    def test_compiler_injects_domain_scope_before_user_filters(self):
        model = SemanticModelDefinition.model_validate(
            km_asset_definition(schema_name="KBOTUI_DEV")
        )
        plan = DataQueryPlanV1.model_validate({
            "contract_version": "DataQueryPlan.v1",
            "semantic_model_id": "01900000-0000-7000-8000-000000000003",
            "semantic_model_version": 1,
            "dataset": "assets",
            "dimensions": ["author"],
            "measures": [{"name": "asset_count", "aggregation": "COUNT"}],
            "filters": [],
            "order_by": [],
            "limit": 100,
        })
        compiled = compile_dialect_query(
            dialect="ORACLE", plan=plan, model=model,
            policy_max_limit=1000, scope_value=41,
        )
        self.assertIn('WHERE "DOMAIN_ID" = :p1', compiled.sql)
        self.assertEqual((41, 100), compiled.parameters)

    def test_raw_metadata_normalization_preserves_business_columns(self):
        normalized = KmAssetService._normalize({"ASSET_TITLE": "AI Asset", "Author_Mail": "u@example.com"})
        columns = KmAssetService._columns(normalized)
        self.assertEqual("AI Asset", columns["asset_title"])
        self.assertEqual("u@example.com", columns["author_mail"])

    def test_km_asset_owns_knowledge_execution_spec(self):
        spec = AgentExecutionSpec(
            schema_version="1.0",
            owner_app_id="km_asset",
            domain_id=1,
            consumer_agent_id=UUID("01900000-0000-7000-8000-000000000001"),
            consumer_agent_version_id=UUID("01900000-0000-7000-8000-000000000002"),
            agent_kind="KNOWLEDGE_RETRIEVAL",
            display_name="KM Agent",
            enabled_capabilities=("conversation", "document", "data_query"),
            models={},
            resource_context={"managed_model": True},
        )
        self.assertEqual("km_asset", spec.owner_app_id)

    def test_active_agent_requires_router_model(self):
        with self.assertRaises(KmAssetApplicationError) as raised:
            KmAgentService._validate_models({"composer_llm": "model-id"})
        self.assertEqual("AGENT_ROUTER_MODEL_REQUIRED", raised.exception.code)

    def test_sharepoint_share_id_uses_decoded_url(self):
        encoded = SharePointClient._share_id(
            "https://example.sharepoint.com/sites/km/Shared%20Documents/a.pdf"
        )
        decoded = SharePointClient._share_id(
            "https://example.sharepoint.com/sites/km/Shared Documents/a.pdf"
        )
        self.assertEqual(decoded, encoded)

    def test_source_update_requires_change_and_concurrency_version(self):
        with self.assertRaises(ValidationError):
            SourceUpdateRequest(domain_id=1, expected_row_version=1)
        payload = SourceUpdateRequest(
            domain_id=1,
            expected_row_version=3,
            poll_interval_seconds=300,
            batch_size=250,
        )
        self.assertEqual(300, payload.poll_interval_seconds)
        self.assertEqual(250, payload.batch_size)

    def test_source_update_rejects_partial_credential_rotation(self):
        with self.assertRaises(ValidationError):
            SourceUpdateRequest(
                domain_id=1,
                expected_row_version=1,
                metadb_credentials={"username": "user", "token": "bad"},
            )


class KmSourceUpdateTest(unittest.IsolatedAsyncioTestCase):
    async def test_update_source_applies_runtime_settings_and_increments_version(self):
        row = SimpleNamespace(
            source_id=UUID("01900000-0000-7000-8000-000000000001"),
            domain_id=1,
            display_name="原来源",
            metadb_endpoint="https://old.example.com/assets",
            metadb_credential_id=UUID("01900000-0000-7000-8000-000000000002"),
            sharepoint_credential_id=UUID("01900000-0000-7000-8000-000000000003"),
            sharepoint_site_path="/sites/old",
            collection_id=UUID("01900000-0000-7000-8000-000000000004"),
            semantic_model_id=None,
            policy_binding_id=None,
            model_status="READY",
            model_catalog_hash="hash",
            status="ACTIVE",
            poll_interval_seconds=60,
            batch_size=100,
            last_sync_at=None,
            error_code=None,
            error_message=None,
            row_version=2,
            updated_by="old-user",
        )

        class Assets:
            async def get_source(self, **_):
                return row

        class Uow:
            assets = Assets()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return None

            async def commit(self):
                return None

        service = KmAssetService(
            uow_factory=Uow,
            credential_service=SimpleNamespace(),
        )
        result = await service.update_source(
            domain_id=1,
            source_id=row.source_id,
            expected_row_version=2,
            changes={"poll_interval_seconds": 300, "batch_size": 250},
            actor_id="kbotui_dev",
        )
        self.assertEqual(300, result["poll_interval_seconds"])
        self.assertEqual(250, result["batch_size"])
        self.assertEqual(3, result["row_version"])
        self.assertEqual("kbotui_dev", row.updated_by)


if __name__ == "__main__":
    unittest.main()
