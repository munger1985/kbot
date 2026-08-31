"""KM Asset 固定问数模型与执行边界测试。"""

import json
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import UUID

from pydantic import ValidationError

from km_asset_app.application import KmAgentService, KmAssetApplicationError, KmAssetService
from km_asset_app.application.agents import KM_AGENT_CAPABILITIES
from km_asset_app.application.worker import (
    KmAssetWorker,
    _JobSnapshot,
    _KcPublicationFailed,
    _KcRevisionPending,
    _SourceSnapshot,
)
from km_asset_app.integrations import (
    SharePointClient,
    SharePointDownloadError,
    SharePointFile,
)
from km_asset_app.api.agents import AgentCreateRequest
from km_asset_app.api.assets import SourceUpdateRequest
from data_query.application.managed_datasets import km_asset_definition
from data_query.contracts import DataQueryPlanV1, SemanticModelDefinition
from data_query.connectors import compile_dialect_query
from platform_core.contracts import AgentExecutionSpec


class KmAssetCatalogTest(unittest.TestCase):
    def test_internal_agent_create_rejects_caller_supplied_capabilities(self):
        with self.assertRaises(ValidationError) as raised:
            AgentCreateRequest(
                domain_id=1,
                source_id=UUID("01900000-0000-7000-8000-000000000001"),
                display_name="KM Agent",
                enabled_capabilities=["conversation", "document", "data_query"],
            )

        self.assertEqual("extra_forbidden", raised.exception.errors()[0]["type"])
        self.assertEqual(("enabled_capabilities",), raised.exception.errors()[0]["loc"])

    def test_agent_uses_fixed_document_and_data_capabilities(self):
        self.assertEqual(("document", "data_query"), KM_AGENT_CAPABILITIES)

    def test_catalog_is_managed_and_domain_scoped(self):
        model = SemanticModelDefinition.model_validate(
            km_asset_definition(schema_name="KBOTUI_DEV")
        )
        self.assertEqual("KBOT_V_KM_ASSET_SEARCHABLE", model.datasets[0].physical_object)
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

    def test_author_filter_matches_normalized_email_and_local_part(self):
        model = SemanticModelDefinition.model_validate(
            km_asset_definition(schema_name="KBOTUI_DEV")
        )
        plan = DataQueryPlanV1.model_validate({
            "semantic_model_id": "01900000-0000-7000-8000-000000000003",
            "semantic_model_version": 1,
            "dataset": "assets",
            "dimensions": ["asset_id", "title", "author"],
            "measures": [{"name": "asset_count", "aggregation": "COUNT"}],
            "filters": [{
                "field": "author",
                "operator": "EQ",
                "values": ["  Lavkesh.Singh@Oracle.COM  "],
            }],
            "limit": 100,
        })

        compiled = compile_dialect_query(
            dialect="ORACLE", plan=plan, model=model,
            policy_max_limit=1000, scope_value=41,
        )

        self.assertIn(
            '("AUTHOR_MAIL_NORM" = :p2 OR "AUTHOR_LOCAL_PART" = :p3)',
            compiled.sql,
        )
        self.assertEqual(
            (41, "lavkesh.singh@oracle.com", "lavkesh.singh@oracle.com", 100),
            compiled.parameters,
        )

    def test_semantic_topic_is_not_exposed_as_query_dimension(self):
        model = SemanticModelDefinition.model_validate(
            km_asset_definition(schema_name="KBOTUI_DEV")
        )
        dimensions = {item.name: item for item in model.dimensions}
        self.assertEqual("KC_BUNDLE_ID", dimensions["bundle_id"].physical_column)
        self.assertEqual(
            "KC_BUNDLE_REVISION_ID",
            dimensions["bundle_revision_id"].physical_column,
        )
        self.assertNotIn("topic", {item.name for item in model.dimensions})

        self.assertNotIn("topic", dimensions)

    def test_asset_date_filter_uses_typed_date_parameter(self):
        model = SemanticModelDefinition.model_validate(
            km_asset_definition(schema_name="KBOTUI_DEV")
        )
        plan = DataQueryPlanV1.model_validate({
            "semantic_model_id": "01900000-0000-7000-8000-000000000003",
            "semantic_model_version": 1,
            "dataset": "assets",
            "dimensions": ["asset_id", "title", "asset_date"],
            "measures": [{"name": "asset_count", "aggregation": "COUNT"}],
            "filters": [{
                "field": "asset_date",
                "operator": "GTE",
                "values": ["2026-01-01"],
            }],
            "limit": 100,
        })

        compiled = compile_dialect_query(
            dialect="ORACLE", plan=plan, model=model,
            policy_max_limit=1000, scope_value=41,
        )

        self.assertIn('"ASSET_DATE_VALUE" >= :p2', compiled.sql)
        self.assertEqual((41, date(2026, 1, 1), 100), compiled.parameters)

    def test_oracle_schema_defines_indexed_author_and_effective_date_columns(self):
        schema = (
            Path(__file__).resolve().parents[3]
            / "database/oracle/km_asset_app/001_km_asset.sql"
        ).read_text(encoding="utf-8")
        self.assertIn("AUTHOR_MAIL_NORM GENERATED ALWAYS AS", schema)
        self.assertIn("AUTHOR_LOCAL_PART GENERATED ALWAYS AS", schema)
        self.assertIn("ASSET_DATE_VALUE GENERATED ALWAYS AS", schema)
        self.assertIn("SUBSTR(TRIM(LAST_UPDATE_TIME), 1, 10)", schema)
        self.assertNotIn(
            "TRUNC(CAST(SYS_EXTRACT_UTC(CREATED_AT) AS DATE))", schema
        )
        self.assertIn("IX_KM_ASSET_AUTHOR_LOCAL", schema)
        self.assertIn("IX_KM_ASSET_DATE", schema)

    def test_raw_metadata_normalization_preserves_business_columns(self):
        normalized = KmAssetService._normalize({"ASSET_TITLE": "AI Asset", "Author_Mail": "u@example.com"})
        columns = KmAssetService._columns(normalized)
        self.assertEqual("AI Asset", columns["asset_title"])
        self.assertEqual("u@example.com", columns["author_mail"])

    def test_source_date_uses_same_priority_as_manifest(self):
        normalized = KmAssetService._normalize({
            "PUBLISH_TIME": "2025-03-04T08:00:00Z",
            "LAST_UPDATE_TIME": "2025-04-05T09:00:00Z",
            "PUBLISH_DATE": "2025-02-03",
        })

        columns = KmAssetService._columns(normalized)

        self.assertEqual(
            "2025-03-04T08:00:00Z", columns["publish_date"]
        )
        self.assertEqual(
            "2025-04-05T09:00:00Z", columns["last_update_time"]
        )

    def test_source_date_falls_back_without_using_ingestion_time(self):
        columns = KmAssetService._columns(
            KmAssetService._normalize({
                "LAST_UPDATE_TIME": "2025-04-05T09:00:00Z",
            })
        )
        missing = KmAssetService._columns({})

        self.assertEqual(
            "2025-04-05T09:00:00Z", columns["publish_date"]
        )
        self.assertIsNone(missing["publish_date"])

    def test_km_asset_owns_knowledge_execution_spec(self):
        spec = AgentExecutionSpec(
            schema_version="1.0",
            owner_app_id="km_asset",
            domain_id=1,
            consumer_agent_id=UUID("01900000-0000-7000-8000-000000000001"),
            consumer_agent_version_id=UUID("01900000-0000-7000-8000-000000000002"),
            agent_kind="KNOWLEDGE_RETRIEVAL",
            display_name="KM Agent",
            enabled_capabilities=KM_AGENT_CAPABILITIES,
            models={},
            resource_context={"managed_model": True},
        )
        self.assertEqual("km_asset", spec.owner_app_id)
        self.assertEqual(("document", "data_query"), spec.enabled_capabilities)
        self.assertNotIn("conversation", spec.enabled_capabilities)

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

    def test_sharepoint_site_candidates_accept_full_config_and_url_fallback(self):
        candidates = SharePointClient._site_path_candidates(
            source_url=(
                "https://example.sharepoint.com/:p:/r/sites/actual/"
                "Shared%20Documents/demo.pdf"
            ),
            configured_site_path=SharePointClient._normalize_site_path(
                "https://example.sharepoint.com/sites/configured/"
            ),
        )

        self.assertEqual(
            ("/sites/configured", "/sites/actual"),
            candidates,
        )

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


class KmAgentBindingTest(unittest.IsolatedAsyncioTestCase):
    async def test_sharepoint_path_download_falls_back_to_url_site(self):
        source_url = (
            "https://example.sharepoint.com/:p:/r/sites/actual/"
            "Shared%20Documents/demo.pdf?d=token"
        )

        class Response:
            def __init__(self, *, status, payload=None, content=b""):
                self.status = status
                self._payload = payload or {}
                self._content = content

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return None

            async def json(self, **_):
                return self._payload

            async def read(self):
                return self._content

        class Session:
            def __init__(self):
                self.urls = []

            def get(self, url, **_):
                self.urls.append(url)
                if url.endswith(
                    "/sites/example.sharepoint.com:/sites/configured"
                ):
                    return Response(
                        status=404,
                        payload={"error": {"code": "itemNotFound"}},
                    )
                if url.endswith(
                    "/sites/example.sharepoint.com:/sites/actual"
                ):
                    return Response(
                        status=200,
                        payload={"id": "actual-site"},
                    )
                if url.endswith(":/content"):
                    return Response(status=200, content=b"demo")
                return Response(
                    status=200,
                    payload={
                        "id": "document-id",
                        "name": "demo.pdf",
                        "file": {"mimeType": "application/pdf"},
                    },
                )

        client = SharePointClient(
            tenant_id="tenant",
            client_id="client",
            client_secret="secret",
            site_path="/sites/configured",
        )
        session = Session()

        metadata, content = await client._download_by_path(
            session=session,
            headers={},
            source_url=source_url,
        )

        self.assertEqual("document-id", metadata["id"])
        self.assertEqual(b"demo", content)
        self.assertTrue(any("/sites/configured" in url for url in session.urls))
        self.assertTrue(any("/sites/actual" in url for url in session.urls))

    async def test_update_agent_creates_new_version_and_keeps_active_status(self):
        agent_id = UUID("01900000-0000-7000-8000-000000000061")
        source_id = UUID("01900000-0000-7000-8000-000000000062")
        collection_id = UUID("01900000-0000-7000-8000-000000000063")
        semantic_model_id = UUID("01900000-0000-7000-8000-000000000064")
        policy_binding_id = UUID("01900000-0000-7000-8000-000000000065")
        router_model_id = UUID("01900000-0000-7000-8000-000000000066")
        agent = SimpleNamespace(
            agent_id=agent_id,
            domain_id=43,
            display_name="旧名称",
            description=None,
            status="ACTIVE",
            current_version_id=UUID(
                "01900000-0000-7000-8000-000000000067"
            ),
            row_version=3,
            updated_by="old-user",
        )
        source = SimpleNamespace(
            model_status="READY",
            semantic_model_id=semantic_model_id,
            policy_binding_id=policy_binding_id,
            collection_id=collection_id,
        )
        versions = []

        class Agents:
            async def get(self, **_):
                return agent

            async def next_version_no(self, **_):
                return 2

            async def add(self, value):
                versions.append(value)

            async def version(self, *, version_id, **_):
                return next(
                    (
                        item
                        for item in versions
                        if item.agent_version_id == version_id
                    ),
                    None,
                )

        class Assets:
            async def get_source(self, **_):
                return source

        class Uow:
            agents = Agents()
            assets = Assets()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return None

            async def commit(self):
                return None

        data_query = SimpleNamespace(management_create=AsyncMock())
        knowledge_core = SimpleNamespace(bind_collection=AsyncMock())
        service = KmAgentService(
            uow_factory=Uow,
            data_query_client=data_query,
            knowledge_core_client=knowledge_core,
        )

        result = await service.update(
            domain_id=43,
            agent_id=agent_id,
            expected_row_version=3,
            source_id=source_id,
            display_name="新名称",
            description="新的描述",
            models={"router_llm": router_model_id},
            instruction="只回答 KM Asset 问题",
            actor_id="kmadmin",
        )

        self.assertEqual(1, len(versions))
        self.assertEqual(2, versions[0].version_no)
        self.assertEqual(versions[0].agent_version_id, agent.current_version_id)
        self.assertEqual("ACTIVE", agent.status)
        self.assertEqual(4, agent.row_version)
        self.assertEqual("新名称", result["display_name"])
        binding = data_query.management_create.await_args.kwargs["payload"]
        self.assertEqual("km_asset", binding["consumer_app_id"])
        self.assertEqual(
            str(versions[0].agent_version_id), binding["agent_version_id"]
        )
        knowledge_core.bind_collection.assert_awaited_once()

    async def test_update_agent_restores_previous_version_when_binding_fails(self):
        agent_id = UUID("01900000-0000-7000-8000-000000000071")
        old_version_id = UUID("01900000-0000-7000-8000-000000000072")
        source_id = UUID("01900000-0000-7000-8000-000000000073")
        agent = SimpleNamespace(
            agent_id=agent_id,
            domain_id=43,
            display_name="可用版本",
            description="旧描述",
            status="ACTIVE",
            current_version_id=old_version_id,
            row_version=5,
            updated_by="old-user",
        )
        source = SimpleNamespace(
            model_status="READY",
            semantic_model_id=UUID(
                "01900000-0000-7000-8000-000000000074"
            ),
            policy_binding_id=UUID(
                "01900000-0000-7000-8000-000000000075"
            ),
            collection_id=UUID(
                "01900000-0000-7000-8000-000000000076"
            ),
        )
        versions = []

        class Agents:
            async def get(self, **_):
                return agent

            async def next_version_no(self, **_):
                return 2

            async def add(self, value):
                versions.append(value)

        class Assets:
            async def get_source(self, **_):
                return source

        class Uow:
            agents = Agents()
            assets = Assets()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return None

            async def commit(self):
                return None

        service = KmAgentService(
            uow_factory=Uow,
            data_query_client=SimpleNamespace(
                management_create=AsyncMock(
                    side_effect=RuntimeError("Data Query 不可用")
                )
            ),
            knowledge_core_client=SimpleNamespace(
                bind_collection=AsyncMock()
            ),
        )

        with self.assertRaisesRegex(RuntimeError, "Data Query 不可用"):
            await service.update(
                domain_id=43,
                agent_id=agent_id,
                expected_row_version=5,
                source_id=source_id,
                display_name="失败的新名称",
                description="新描述",
                models={
                    "router_llm": UUID(
                        "01900000-0000-7000-8000-000000000077"
                    )
                },
                instruction=None,
                actor_id="kmadmin",
            )

        self.assertEqual(old_version_id, agent.current_version_id)
        self.assertEqual("可用版本", agent.display_name)
        self.assertEqual("旧描述", agent.description)
        self.assertEqual("ACTIVE", agent.status)
        self.assertEqual(7, agent.row_version)

    async def test_execution_spec_fixes_km_security_level(self):
        service = KmAgentService(
            uow_factory=None,
            data_query_client=None,
            knowledge_core_client=SimpleNamespace(bind_collection=AsyncMock()),
        )
        service.get = AsyncMock(return_value={
            "status": "ACTIVE",
            "agent_id": "01900000-0000-7000-8000-000000000001",
            "agent_version_id": "01900000-0000-7000-8000-000000000002",
            "collection_id": "01900000-0000-7000-8000-000000000003",
            "semantic_model_id": "01900000-0000-7000-8000-000000000004",
            "policy_binding_id": "01900000-0000-7000-8000-000000000005",
            "source_id": "01900000-0000-7000-8000-000000000006",
            "display_name": "KM Agent",
            "models": {"router_llm": {}},
            "instruction": None,
            "config": {},
        })
        service._ensure_collection_binding = AsyncMock()

        result = await service.execution_spec(
            domain_id=43,
            agent_id=UUID("01900000-0000-7000-8000-000000000001"),
            actor_id="kmadmin",
        )

        self.assertNotIn("security_level", result["resource_context"])

    async def test_km_agent_ensures_kc_collection_binding(self):
        knowledge_core = SimpleNamespace(bind_collection=AsyncMock())
        service = KmAgentService(
            uow_factory=None,
            data_query_client=None,
            knowledge_core_client=knowledge_core,
        )
        agent_id = UUID("01900000-0000-7000-8000-000000000001")
        collection_id = UUID("01900000-0000-7000-8000-000000000002")

        await service._ensure_collection_binding(
            domain_id=43,
            agent_id=agent_id,
            collection_id=collection_id,
            actor_id="kmadmin",
        )

        call = knowledge_core.bind_collection.await_args.kwargs
        self.assertEqual(43, call["domain_id"])
        self.assertEqual(agent_id, call["agent_id"])
        self.assertEqual(collection_id, call["collection_id"])
        self.assertEqual("43", call["auth_context"].domain_id)
        self.assertEqual("kmadmin", call["auth_context"].asserted_user_id)


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


class KmAssetReindexTest(unittest.IsolatedAsyncioTestCase):
    async def test_reindex_persists_local_status_tracking_job(self):
        asset_id = UUID("01900000-0000-7000-8000-000000000031")
        source_id = UUID("01900000-0000-7000-8000-000000000032")
        collection_id = UUID("01900000-0000-7000-8000-000000000033")
        bundle_id = UUID("01900000-0000-7000-8000-000000000034")
        revision_id = UUID("01900000-0000-7000-8000-000000000035")
        asset = SimpleNamespace(
            km_asset_id=asset_id,
            source_id=source_id,
            current_revision_id=UUID(
                "01900000-0000-7000-8000-000000000038"
            ),
            external_asset_id="ASSET-1",
            source_revision="1",
            source_status="Y",
            ingestion_status="READY",
            asset_title="ChatBI",
            author_mail="author@example.com",
            asset_product=None,
            asset_solution="ChatBI",
            industry_id=None,
            content_category=None,
            asset_status="Published",
            publish_date=None,
            last_update_time=None,
            kc_bundle_id=bundle_id,
            kc_bundle_revision_id=revision_id,
            failure_stage=None,
            error_code=None,
            error_message=None,
            attempt_count=0,
            synced_at=None,
            completed_at=None,
            row_version=2,
        )
        source = SimpleNamespace(collection_id=collection_id)
        uow_entries = 0
        added_jobs = []

        class Assets:
            async def get_asset(self, **_):
                return asset

            async def get_source(self, **_):
                return source

            async def list_latest_reindex_jobs(self, **_):
                return []

            async def add(self, job):
                added_jobs.append(job)

        class Uow:
            assets = Assets()

            async def __aenter__(self):
                nonlocal uow_entries
                uow_entries += 1
                return self

            async def __aexit__(self, *_):
                return None

            async def commit(self):
                return None

        receipt = {
            "generation": "01900000-0000-7000-8000-000000000036",
            "profile_job_id": "01900000-0000-7000-8000-000000000037",
            "expected_bundle_row_version": 4,
        }
        knowledge_core = SimpleNamespace(
            reindex_discovery=AsyncMock(return_value=receipt)
        )
        service = KmAssetService(
            uow_factory=Uow,
            credential_service=SimpleNamespace(),
            knowledge_core_client=knowledge_core,
        )

        result = await service.reindex_asset(
            domain_id=43,
            km_asset_id=asset_id,
            expected_row_version=2,
            actor_id="kmadmin",
        )

        self.assertEqual(2, uow_entries)
        self.assertEqual("PENDING", result["status"])
        self.assertEqual(receipt, result["kc_reindex"])
        self.assertEqual("PENDING", result["tracking_status"])
        self.assertEqual(1, len(added_jobs))
        self.assertEqual("KC_STATUS_SYNC", added_jobs[0].job_type)
        self.assertEqual(
            "DISCOVERY_REINDEX",
            added_jobs[0].payload_json["operation_type"],
        )
        self.assertFalse(added_jobs[0].payload_json["recover_asset"])
        self.assertEqual("READY", result["asset"]["ingestion_status"])
        knowledge_core.reindex_discovery.assert_awaited_once()

    def test_kc_accepted_asset_requires_reindex_recovery(self):
        self.assertTrue(KmAssetService._requires_reindex_recovery({
            "ingestion_status": "KC_ACCEPTED",
            "source_status": "N",
        }))
        self.assertFalse(KmAssetService._requires_reindex_recovery({
            "ingestion_status": "READY",
            "source_status": "Y",
        }))

    async def test_batch_reindex_keeps_partial_success_results(self):
        first_id = UUID("01900000-0000-7000-8000-000000000071")
        second_id = UUID("01900000-0000-7000-8000-000000000072")
        service = KmAssetService(
            uow_factory=SimpleNamespace(),
            credential_service=SimpleNamespace(),
        )
        service.reindex_asset = AsyncMock(side_effect=[
            {
                "status": "PENDING",
                "kc_reindex": {"profile_job_id": "job-1"},
                "tracking_status": "PENDING",
                "tracking_job": {"job_id": "tracking-1"},
            },
            KmAssetApplicationError(
                status_code=409,
                code="ROW_VERSION_CONFLICT",
                message="Asset 已被其他请求修改",
            ),
        ])

        result = await service.batch_reindex_assets(
            domain_id=43,
            items=[
                {"km_asset_id": first_id, "expected_row_version": 2},
                {"km_asset_id": second_id, "expected_row_version": 3},
            ],
            actor_id="kmadmin",
        )

        self.assertEqual(2, result["requested_count"])
        self.assertEqual(1, result["submitted_count"])
        self.assertEqual(1, result["failed_count"])
        self.assertEqual(0, result["untracked_count"])
        self.assertEqual("SUBMITTED", result["results"][0]["status"])
        self.assertEqual(
            "ROW_VERSION_CONFLICT",
            result["results"][1]["error_code"],
        )


class KmWorkerSnapshotTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _kc_job() -> _JobSnapshot:
        return _JobSnapshot(
            job_id=UUID("01900000-0000-7000-8000-000000000041"),
            job_type="KC_STATUS_SYNC",
            domain_id=41,
            source_id=UUID("01900000-0000-7000-8000-000000000042"),
            km_asset_id=UUID("01900000-0000-7000-8000-000000000043"),
            asset_revision_id=UUID("01900000-0000-7000-8000-000000000044"),
            payload_json={
                "bundle_id": "01900000-0000-7000-8000-000000000045",
                "bundle_revision_id": "01900000-0000-7000-8000-000000000046",
            },
        )

    async def test_failed_attachment_keeps_metadata_bundle_ingestable(self):
        source_id = UUID("01900000-0000-7000-8000-000000000051")
        asset_id = UUID("01900000-0000-7000-8000-000000000052")
        revision_id = UUID("01900000-0000-7000-8000-000000000053")
        collection_id = UUID("01900000-0000-7000-8000-000000000054")
        bundle_id = UUID("01900000-0000-7000-8000-000000000055")
        kc_revision_id = UUID("01900000-0000-7000-8000-000000000056")
        bad_url = "https://example.sharepoint.com/broken/ChatBI.pdf"
        good_url = "https://example.sharepoint.com/files/Overview.pdf"
        duplicate_good_url = (
            "https://example.sharepoint.com/shared/Overview.pdf"
        )
        asset = SimpleNamespace(
            km_asset_id=asset_id,
            external_asset_id="ASSET-1",
            asset_title="ChatBI Asset",
            raw_metadata_json={
                "first_sp_url": (
                    f"{bad_url}^^^{good_url}^^^{duplicate_good_url}"
                ),
                "second_sp_url": bad_url,
            },
            normalized_metadata_json={"asset_title": "ChatBI Asset"},
            ingestion_status="SYNC_PENDING",
            kc_bundle_id=None,
            kc_bundle_revision_id=None,
            failure_stage="ATTACHMENT_DOWNLOAD",
            error_code="OLD_ERROR",
            error_message="旧错误",
        )
        source = SimpleNamespace(
            source_id=source_id,
            sharepoint_credential_id=UUID(
                "01900000-0000-7000-8000-000000000057"
            ),
            sharepoint_site_path="/sites/km",
            collection_id=collection_id,
        )
        revision = SimpleNamespace(
            source_revision="1",
            snapshot_hash="a" * 64,
            status="DISCOVERED",
            kc_bundle_revision_id=None,
        )
        added = []
        attachments = {}

        class Assets:
            async def get_asset(self, **_):
                return asset

            async def get_source(self, **_):
                return source

            async def get_revision(self, **_):
                return revision

            async def find_attachment(
                self, *, asset_revision_id, external_document_id
            ):
                return attachments.get(external_document_id)

            async def add(self, value):
                added.append(value)
                if hasattr(value, "external_document_id"):
                    attachments[value.external_document_id] = value

        class Uow:
            assets = Assets()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return None

            async def commit(self):
                return None

        class Credentials:
            async def read(self, **_):
                return {
                    "tenant_id": "tenant",
                    "client_id": "client",
                    "client_secret": "secret",
                }

        download_calls = []

        class SharePoint:
            def __init__(self, **_):
                pass

            async def download(self, source_url):
                download_calls.append(source_url)
                if source_url == bad_url:
                    raise SharePointDownloadError("附件地址不存在")
                return SharePointFile(
                    external_document_id="drive-item-good",
                    name="Overview.pdf",
                    mime_type="application/pdf",
                    content=b"valid attachment",
                )

        class MultipartPart:
            def __init__(self, owner, value):
                self._owner = owner
                self._value = value

            def set_content_disposition(self, _, *, name, **__):
                self._owner.values[name] = self._value

        class Multipart:
            def __init__(self, _):
                self.content_type = "multipart/form-data; boundary=test"
                self.values = {}

            def append(self, value, _headers=None):
                return MultipartPart(self, value)

        class KnowledgeCore:
            request = None

            async def ingest_multipart(self, **values):
                self.request = values
                return SimpleNamespace(payload={
                    "bundle_id": str(bundle_id),
                    "bundle_revision_id": str(kc_revision_id),
                })

        knowledge_core = KnowledgeCore()
        worker = KmAssetWorker(
            uow_factory=Uow,
            credential_service=Credentials(),
            asset_service=SimpleNamespace(),
            knowledge_core_client=knowledge_core,
        )
        job = _JobSnapshot(
            job_id=UUID("01900000-0000-7000-8000-000000000058"),
            job_type="ATTACHMENT_DOWNLOAD",
            domain_id=41,
            source_id=source_id,
            km_asset_id=asset_id,
            asset_revision_id=revision_id,
            payload_json={},
        )

        with (
            patch(
                "km_asset_app.application.worker.SharePointClient",
                SharePoint,
            ),
            patch(
                "km_asset_app.application.worker.aiohttp.MultipartWriter",
                Multipart,
            ),
        ):
            await worker._download_and_ingest(job)

        parts = knowledge_core.request["body"].values
        documents = json.loads(parts["documents"])
        self.assertEqual(1, len(documents))
        self.assertEqual("drive-item-good", documents[0]["external_document_id"])
        failures = json.loads(parts["document_failures"])
        self.assertEqual(1, len(failures))
        self.assertEqual("SOURCE_DOWNLOAD_FAILED", failures[0]["failure_code"])
        self.assertEqual(bad_url, failures[0]["source_url"])
        self.assertEqual(
            [bad_url, good_url, duplicate_good_url],
            download_calls,
        )
        failed_attachment = next(
            item
            for item in added
            if getattr(item, "status", None) == "FAILED"
        )
        self.assertEqual("FAILED", failed_attachment.status)
        self.assertTrue(
            any(getattr(item, "status", None) == "AVAILABLE" for item in added)
        )
        self.assertEqual("KC_ACCEPTED", asset.ingestion_status)
        self.assertIsNone(asset.failure_stage)
        self.assertIsNone(asset.error_code)
        self.assertEqual("PROCESSING", revision.status)
        self.assertTrue(
            any(getattr(item, "job_type", None) == "KC_STATUS_SYNC" for item in added)
        )

    async def test_kc_status_sync_tracks_exact_revision(self):
        asset = SimpleNamespace(
            ingestion_status="PARSING",
            completed_at=None,
            external_asset_id="ASSET-1",
            source_status="N",
            failure_stage="KC_STATUS_SYNC",
            error_code="OLD_ERROR",
            error_message="旧错误",
            row_version=1,
        )
        revision = SimpleNamespace(status="PROCESSING")
        added = []

        class Assets:
            async def get_asset(self, **_):
                return asset

            async def get_revision(self, **_):
                return revision

            async def find_job_by_key(self, **_):
                return None

            async def add(self, value):
                added.append(value)

        class Uow:
            assets = Assets()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return None

            async def commit(self):
                return None

        class KnowledgeCore:
            request = None

            async def get_revision_status(self, **values):
                self.request = values
                return {
                    "status": "PARTIAL",
                    "publication_status": "PUBLISHED",
                }

        kc = KnowledgeCore()
        worker = KmAssetWorker(
            uow_factory=Uow,
            credential_service=SimpleNamespace(),
            asset_service=SimpleNamespace(),
            knowledge_core_client=kc,
        )
        await worker._kc_status_sync(self._kc_job())

        self.assertEqual(
            UUID("01900000-0000-7000-8000-000000000046"),
            kc.request["bundle_revision_id"],
        )
        self.assertFalse(kc.request["include_members"])
        self.assertEqual("READY", asset.ingestion_status)
        self.assertEqual("READY", revision.status)
        self.assertIsNone(asset.failure_stage)
        self.assertIsNone(asset.error_code)
        self.assertIsNone(asset.error_message)
        self.assertEqual("SOURCE_STATUS_UPDATE", added[0].job_type)

    async def test_kc_status_sync_waits_for_discovery_publication(self):
        class KnowledgeCore:
            async def get_revision_status(self, **_):
                return {
                    "status": "READY",
                    "publication_status": "PUBLISHING",
                }

        worker = KmAssetWorker(
            uow_factory=SimpleNamespace(),
            credential_service=SimpleNamespace(),
            asset_service=SimpleNamespace(),
            knowledge_core_client=KnowledgeCore(),
        )

        with self.assertRaisesRegex(
            _KcRevisionPending, "DISCOVERY_PUBLISHING"
        ):
            await worker._kc_status_sync(self._kc_job())

    async def test_kc_status_sync_stops_on_discovery_publication_failure(self):
        class KnowledgeCore:
            async def get_revision_status(self, **_):
                return {
                    "status": "PARTIAL",
                    "publication_status": "FAILED",
                    "publication_failure_code": "WORKER_RUN_FAILED",
                    "publication_failure_message": "Embedding 输入过长",
                }

        worker = KmAssetWorker(
            uow_factory=SimpleNamespace(),
            credential_service=SimpleNamespace(),
            asset_service=SimpleNamespace(),
            knowledge_core_client=KnowledgeCore(),
        )

        with self.assertRaisesRegex(
            _KcPublicationFailed, "WORKER_RUN_FAILED",
        ):
            await worker._kc_status_sync(self._kc_job())

    async def test_kc_status_sync_completes_superseded_revision_without_write(self):
        class KnowledgeCore:
            async def get_revision_status(self, **_):
                return {
                    "status": "READY",
                    "publication_status": "SUPERSEDED",
                }

        worker = KmAssetWorker(
            uow_factory=SimpleNamespace(),
            credential_service=SimpleNamespace(),
            asset_service=SimpleNamespace(),
            knowledge_core_client=KnowledgeCore(),
        )

        await worker._kc_status_sync(self._kc_job())

    def test_pending_revision_log_is_limited_until_five_minutes(self):
        worker = KmAssetWorker(
            uow_factory=SimpleNamespace(),
            credential_service=SimpleNamespace(),
            asset_service=SimpleNamespace(),
            knowledge_core_client=SimpleNamespace(),
        )
        job_id = UUID("01900000-0000-7000-8000-000000000041")

        with (
            patch(
                "km_asset_app.application.worker.time.monotonic",
                side_effect=[100.0, 101.0, 401.0],
            ),
            patch("km_asset_app.application.worker.logger.debug") as debug,
        ):
            worker._log_pending(job_id, "DISCOVERY_PUBLISHING")
            worker._log_pending(job_id, "DISCOVERY_PUBLISHING")
            worker._log_pending(job_id, "DISCOVERY_PUBLISHING")

        self.assertEqual(2, debug.call_count)

    async def test_processing_kc_revision_is_deferred_without_failure(self):
        class KnowledgeCore:
            async def get_revision_status(self, **_):
                return {"status": "PROCESSING"}

        worker = KmAssetWorker(
            uow_factory=SimpleNamespace(),
            credential_service=SimpleNamespace(),
            asset_service=SimpleNamespace(),
            knowledge_core_client=KnowledgeCore(),
        )
        with self.assertRaises(_KcRevisionPending):
            await worker._kc_status_sync(self._kc_job())

    async def test_successful_job_clears_previous_error(self):
        job = SimpleNamespace(
            status="RUNNING",
            completed_at=None,
            error_code="DetachedInstanceError",
            error_message="旧错误",
            lease_owner="worker",
            lease_until=object(),
        )

        class Assets:
            async def get_job_by_id(self, **_):
                return job

        class Uow:
            assets = Assets()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return None

            async def commit(self):
                return None

        worker = KmAssetWorker(
            uow_factory=Uow,
            credential_service=SimpleNamespace(),
            asset_service=SimpleNamespace(),
            knowledge_core_client=SimpleNamespace(),
        )
        await worker._complete(self._kc_job().job_id, succeeded=True)

        self.assertEqual("SUCCEEDED", job.status)
        self.assertIsNone(job.error_code)
        self.assertIsNone(job.error_message)

    async def test_terminal_failure_reuses_existing_source_status_job(self):
        revision_id = UUID("01900000-0000-7000-8000-000000000047")
        job = SimpleNamespace(
            status="RUNNING",
            attempt_count=5,
            max_attempts=5,
            completed_at=None,
            error_code=None,
            error_message=None,
            lease_owner="worker",
            lease_until=object(),
            domain_id=41,
            source_id=UUID("01900000-0000-7000-8000-000000000031"),
            km_asset_id=UUID("01900000-0000-7000-8000-000000000048"),
            asset_revision_id=revision_id,
            job_type="RETRY",
            payload_json={},
        )
        asset = SimpleNamespace(
            external_asset_id="ASSET-1",
            ingestion_status="READY",
            failure_stage=None,
            error_code=None,
            error_message=None,
            row_version=1,
        )

        class Assets:
            added = []
            requested_key = None

            async def get_job_by_id(self, **_):
                return job

            async def get_asset(self, **_):
                return asset

            async def find_job_by_key(self, **values):
                self.requested_key = values["idempotency_key"]
                return SimpleNamespace(job_id=UUID(int=1))

            async def add(self, row):
                self.added.append(row)

        assets = Assets()

        class Uow:
            async def __aenter__(self):
                self.assets = assets
                return self

            async def __aexit__(self, *_):
                return None

            async def commit(self):
                return None

        worker = KmAssetWorker(
            uow_factory=Uow,
            credential_service=SimpleNamespace(),
            asset_service=SimpleNamespace(),
            knowledge_core_client=SimpleNamespace(),
        )
        await worker._complete(
            self._kc_job().job_id,
            succeeded=False,
            error=RuntimeError("重试失败"),
        )

        self.assertEqual("FAILED", job.status)
        self.assertEqual(
            f"source-failed:{revision_id}:RETRY",
            assets.requested_key,
        )
        self.assertEqual([], assets.added)
        self.assertIsNone(job.lease_owner)
        self.assertIsNone(job.lease_until)

    async def test_completion_failure_does_not_escape_worker_loop(self):
        worker = KmAssetWorker(
            uow_factory=SimpleNamespace(),
            credential_service=SimpleNamespace(),
            asset_service=SimpleNamespace(),
            knowledge_core_client=SimpleNamespace(),
        )
        worker._complete = AsyncMock(
            side_effect=RuntimeError("收尾事务失败")
        )

        await worker._complete_safely(
            self._kc_job().job_id,
            succeeded=False,
            error=RuntimeError("任务失败"),
        )

        worker._complete.assert_awaited_once()

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
