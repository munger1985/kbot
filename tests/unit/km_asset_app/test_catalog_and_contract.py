"""KM Asset 固定问数模型与执行边界测试。"""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import UUID

from pydantic import ValidationError

from km_asset_app.application import KmAgentService, KmAssetApplicationError, KmAssetService
from km_asset_app.application.agents import KM_AGENT_CAPABILITIES
from km_asset_app.application.worker import (
    KmAssetWorker,
    _JobSnapshot,
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
    async def test_reindex_returns_kc_receipt_without_local_status_job(self):
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

        class Assets:
            async def get_asset(self, **_):
                return asset

            async def get_source(self, **_):
                return source

            async def add(self, _):
                raise AssertionError("重新索引不应创建 KM 本地状态 Job")

        class Uow:
            assets = Assets()

            async def __aenter__(self):
                nonlocal uow_entries
                uow_entries += 1
                return self

            async def __aexit__(self, *_):
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

        self.assertEqual(1, uow_entries)
        self.assertEqual("PENDING", result["status"])
        self.assertEqual(receipt, result["kc_reindex"])
        self.assertEqual("READY", result["asset"]["ingestion_status"])
        knowledge_core.reindex_discovery.assert_awaited_once()


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
        asset = SimpleNamespace(
            km_asset_id=asset_id,
            external_asset_id="ASSET-1",
            asset_title="ChatBI Asset",
            raw_metadata_json={
                "first_sp_url": f"{bad_url}^^^{good_url}"
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

        class SharePoint:
            def __init__(self, **_):
                pass

            async def download(self, source_url):
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
            row_version=1,
        )
        revision = SimpleNamespace(status="PROCESSING")
        added = []

        class Assets:
            async def get_asset(self, **_):
                return asset

            async def get_revision(self, **_):
                return revision

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
                return {"status": "PARTIAL"}

            async def get_bundle_status(self, **_):
                return {
                    "availability_status": "PARTIAL",
                    "current_revision_id": (
                        "01900000-0000-7000-8000-000000000046"
                    ),
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
        self.assertEqual("SOURCE_STATUS_UPDATE", added[0].job_type)

    async def test_kc_status_sync_waits_for_discovery_publication(self):
        class KnowledgeCore:
            async def get_revision_status(self, **_):
                return {"status": "READY"}

            async def get_bundle_status(self, **_):
                return {
                    "availability_status": "PROCESSING",
                    "current_revision_id": None,
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
