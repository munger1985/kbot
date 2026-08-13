"""KM Asset 固定问数模型与执行边界测试。"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID

from pydantic import ValidationError

from km_asset_app.application import KmAgentService, KmAssetApplicationError, KmAssetService
from km_asset_app.application.worker import (
    KmAssetWorker,
    _JobSnapshot,
    _SourceSnapshot,
)
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
            auto_sync_enabled=0,
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
            changes={
                "poll_interval_seconds": 300,
                "batch_size": 250,
                "auto_sync_enabled": True,
            },
            actor_id="kbotui_dev",
        )
        self.assertEqual(300, result["poll_interval_seconds"])
        self.assertEqual(250, result["batch_size"])
        self.assertTrue(result["auto_sync_enabled"])
        self.assertEqual(3, result["row_version"])
        self.assertEqual("kbotui_dev", row.updated_by)

    async def test_metadb_query_does_not_read_source_after_uow_closes(self):
        source_id = UUID("01900000-0000-7000-8000-000000000011")
        credential_id = UUID("01900000-0000-7000-8000-000000000012")

        class Source:
            def __init__(self):
                self.attached = True
                self.metadb_endpoint = "https://metadb.example.com/assets"
                self.metadb_credential_id = credential_id
                self.source_id = source_id

            def __getattribute__(self, name):
                if name not in {"attached", "__class__"} and not object.__getattribute__(self, "attached"):
                    raise RuntimeError("来源实体已脱离 Session")
                return object.__getattribute__(self, name)

        source = Source()

        class Assets:
            async def get_source(self, **_):
                return source

        class Uow:
            assets = Assets()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                source.attached = False

        class Credentials:
            async def read(self, **_):
                return {"username": "user", "password": "secret"}

        class MetaDbClient:
            def __init__(self, **values):
                self.values = values

            async def list_assets(self, **_):
                return [{"asset_id": "A-1"}]

        service = KmAssetService(
            uow_factory=Uow,
            credential_service=Credentials(),
        )
        with patch(
            "km_asset_app.application.assets.AssetMetaDbClient",
            MetaDbClient,
        ):
            result = await service.list_metadb_assets(
                domain_id=1,
                source_id=source_id,
                processed="N",
                offset=0,
                limit=100,
            )

        self.assertEqual([{"asset_id": "A-1"}], result["items"])


class KmWorkerSnapshotTest(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_source_skips_automatic_sync_job(self):
        source_id = UUID("01900000-0000-7000-8000-000000000031")
        worker = KmAssetWorker(
            uow_factory=SimpleNamespace(),
            credential_service=SimpleNamespace(),
            asset_service=SimpleNamespace(),
            knowledge_core_client=SimpleNamespace(),
        )

        async def source_and_credentials(*_):
            return (
                _SourceSnapshot(
                    source_id=source_id,
                    domain_id=41,
                    metadb_endpoint="https://metadb.example.com/assets",
                    batch_size=100,
                    sharepoint_site_path="/sites/km",
                    collection_id=UUID(
                        "01900000-0000-7000-8000-000000000032"
                    ),
                    auto_sync_enabled=False,
                ),
                {"username": "user", "password": "secret"},
            )

        worker._source_and_credentials = source_and_credentials
        await worker._source_sync(
            _JobSnapshot(
                job_id=UUID("01900000-0000-7000-8000-000000000033"),
                job_type="SOURCE_SYNC",
                domain_id=41,
                source_id=source_id,
                km_asset_id=None,
                asset_revision_id=None,
                payload_json={"trigger": "AUTO"},
            )
        )

    async def test_source_credentials_returns_detached_safe_snapshot(self):
        source_id = UUID("01900000-0000-7000-8000-000000000021")
        credential_id = UUID("01900000-0000-7000-8000-000000000022")
        collection_id = UUID("01900000-0000-7000-8000-000000000023")

        class Source:
            def __init__(self):
                self.attached = True
                self.source_id = source_id
                self.domain_id = 41
                self.metadb_endpoint = "https://metadb.example.com/assets"
                self.metadb_credential_id = credential_id
                self.sharepoint_credential_id = credential_id
                self.sharepoint_site_path = "/sites/km"
                self.collection_id = collection_id
                self.batch_size = 100
                self.auto_sync_enabled = 0

            def __getattribute__(self, name):
                if name not in {"attached", "__class__"} and not object.__getattribute__(self, "attached"):
                    raise RuntimeError("来源实体已脱离 Session")
                return object.__getattribute__(self, name)

        source = Source()

        class Assets:
            async def get_source(self, **_):
                return source

        class Uow:
            assets = Assets()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                source.attached = False

        class Credentials:
            async def read(self, **_):
                return {"username": "user", "password": "secret"}

        worker = KmAssetWorker(
            uow_factory=Uow,
            credential_service=Credentials(),
            asset_service=SimpleNamespace(),
            knowledge_core_client=SimpleNamespace(),
        )
        snapshot, values = await worker._source_and_credentials(
            _JobSnapshot(
                job_id=UUID("01900000-0000-7000-8000-000000000024"),
                job_type="SOURCE_SYNC",
                domain_id=41,
                source_id=source_id,
                km_asset_id=None,
                asset_revision_id=None,
                payload_json={},
            ),
            "METADB_BASIC",
        )

        self.assertEqual(source_id, snapshot.source_id)
        self.assertEqual(collection_id, snapshot.collection_id)
        self.assertFalse(snapshot.auto_sync_enabled)
        self.assertEqual({"username": "user", "password": "secret"}, values)


if __name__ == "__main__":
    unittest.main()
