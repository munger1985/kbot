"""运维知识提炼合同与原文命令边界测试。"""

import hashlib
from types import SimpleNamespace
import unittest
from unittest.mock import ANY, AsyncMock

from pydantic import ValidationError

from aiops_agent.application.operations_knowledge import (
    OperationsKnowledgeService,
    _bundle_source_hash,
)
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.contracts.knowledge import (
    KnowledgeAssistantBindingWrite,
    KnowledgeBaseWrite,
    ManualUploadMetadata,
)
from aiops_agent.repositories.knowledge import OperationsKnowledgeRepository
from platform_core.identity import uuid7
from platform_clients.knowledge_core import KnowledgeCoreClientError


async def _body_chunks(body: bytes):
    yield body


class _OrderedFlushSession:
    def __init__(self) -> None:
        self.pending: list[object] = []
        self.flushes: list[list[object]] = []

    def add(self, entity: object) -> None:
        self.pending.append(entity)

    def add_all(self, entities: list[object]) -> None:
        self.pending.extend(entities)

    async def flush(self) -> None:
        self.flushes.append(list(self.pending))
        self.pending.clear()


class OperationsKnowledgeContractTest(unittest.TestCase):
    def test_knowledge_assistant_binding_requires_composer_model(self):
        with self.assertRaises(ValidationError):
            KnowledgeAssistantBindingWrite(
                knowledge_base_ids=(uuid7(),),
            )

    def test_knowledge_base_write_requires_models(self):
        with self.assertRaises(ValidationError):
            KnowledgeBaseWrite(display_name="Oracle 运维库")

    def test_upload_metadata_rejects_internal_collection_fields(self):
        with self.assertRaises(ValidationError):
            ManualUploadMetadata.model_validate({
                "display_name": "Oracle 手册",
                "collection_id": str(uuid7()),
            })

    def test_upload_metadata_rejects_user_selected_scopes(self):
        with self.assertRaises(ValidationError):
            ManualUploadMetadata.model_validate({
                "display_name": "Oracle 手册",
                "scopes": [{"kind": "DATABASE_TYPE", "value": "ORACLE"}],
            })

    def test_bundle_source_hash_preserves_member_order_and_semantics(self):
        documents = [
            {"ordinal": 0, "file_name": "manual.md", "content_sha256": "a" * 64, "role": "CONTENT", "required_flag": True},
            {"ordinal": 1, "file_name": "appendix.txt", "content_sha256": "b" * 64, "role": "SUPPLEMENT", "required_flag": False},
        ]
        digest = _bundle_source_hash(documents)
        self.assertEqual(digest, _bundle_source_hash([dict(item) for item in documents]))
        self.assertNotEqual(digest, _bundle_source_hash(list(reversed(documents))))

    def test_manual_profile_only_emits_commands_present_in_evidence(self):
        asset = SimpleNamespace(asset_id=uuid7(), display_name="归档配置手册")
        version = SimpleNamespace(asset_version_id=uuid7())
        source = SimpleNamespace(source_locator_json={"metadata": {
            "publisher": "DBA Team", "document_version": "1.0",
            "published_date": "2026-09-30",
        }})
        scope = SimpleNamespace(
            scope_kind="DATABASE_TYPE", scope_value="ORACLE"
        )
        profile = OperationsKnowledgeService._manual_profile(
            asset=asset,
            version=version,
            scopes=[scope],
            sources=[source],
            revision={"extraction_evidence": [{
                "evidence_id": str(uuid7()),
                "section_key": "检查归档模式",
                "content_text": (
                    "先确认当前数据库角色。\n"
                    "SELECT LOG_MODE FROM V$DATABASE;\n"
                    "不得根据经验补写其他命令。"
                ),
                "locator": {"page": 3},
            }]},
        )
        self.assertEqual("OPS_MANUAL_PROFILE.v1", profile.schema_version)
        self.assertEqual(("ORACLE",), profile.scope["database_type"])
        self.assertEqual(
            "SELECT LOG_MODE FROM V$DATABASE;",
            profile.procedures[0].steps[0].command_text,
        )
        self.assertEqual(1, len(profile.procedures[0].steps))

    def test_manual_profile_does_not_infer_tags_from_keywords(self):
        profile = OperationsKnowledgeService._manual_profile(
            asset=SimpleNamespace(
                asset_id=uuid7(), display_name="数据库运维手册"
            ),
            version=SimpleNamespace(asset_version_id=uuid7()),
            scopes=[SimpleNamespace(
                scope_kind="DATABASE_TYPE", scope_value="ORACLE"
            )],
            sources=[SimpleNamespace(source_locator_json={"metadata": {}})],
            revision={"extraction_evidence": [{
                "evidence_id": str(uuid7()),
                "content_text": "使用 psql 连接 PostgreSQL 后执行检查。",
                "locator": {"page": 1},
            }]},
        )
        self.assertEqual((), profile.warnings)
        self.assertEqual(("ORACLE",), profile.scope["database_type"])

    def test_manual_profile_preserves_multiline_fenced_command(self):
        command = "ALTER SYSTEM SET LOG_ARCHIVE_CONFIG=\n  'DG_CONFIG=(DB1,DB2)'\n  SCOPE=BOTH;"
        profile = OperationsKnowledgeService._manual_profile(
            asset=SimpleNamespace(asset_id=uuid7(), display_name="DG 手册"),
            version=SimpleNamespace(asset_version_id=uuid7()),
            scopes=[],
            sources=[SimpleNamespace(source_locator_json={"metadata": {}})],
            revision={"extraction_evidence": [{
                "evidence_id": str(uuid7()),
                "content_text": f"```sql\n{command}\n```",
                "locator": {"page": 2},
            }]},
        )
        self.assertEqual(command, profile.procedures[0].steps[0].command_text)


class OperationsKnowledgeRepositoryTest(unittest.IsolatedAsyncioTestCase):
    async def test_asset_version_flushes_parent_rows_before_dependents(self):
        session = _OrderedFlushSession()
        repository = OperationsKnowledgeRepository(session)  # type: ignore[arg-type]
        asset = object()
        version = object()
        scope = object()
        source = object()
        index = object()

        await repository.add_asset_version(
            asset=asset,  # type: ignore[arg-type]
            version=version,  # type: ignore[arg-type]
            scopes=[scope],  # type: ignore[list-item]
            sources=[source],  # type: ignore[list-item]
            indexes=[index],  # type: ignore[list-item]
        )

        self.assertEqual(
            [[asset], [version], [scope, source, index]],
            session.flushes,
        )

    async def test_new_version_flushes_version_before_dependents(self):
        session = _OrderedFlushSession()
        repository = OperationsKnowledgeRepository(session)  # type: ignore[arg-type]
        version = object()
        scope = object()

        await repository.add_version(
            version=version,  # type: ignore[arg-type]
            scopes=[scope],  # type: ignore[list-item]
            sources=[],
            indexes=[],
        )

        self.assertEqual([[version], [scope]], session.flushes)


class OperationsKnowledgeUploadTest(unittest.IsolatedAsyncioTestCase):
    async def test_single_file_upload_preserves_business_display_name(self):
        service = OperationsKnowledgeService(
            uow_factory=lambda: None,
            knowledge_core=SimpleNamespace(),
            knowledge_assistant_id=uuid7(),
            knowledge_assistant_version_id=uuid7(),
            knowledge_assistant_instruction="",
        )
        service.ingest_manual = AsyncMock(return_value={
            "asset": {"asset_id": str(uuid7())},
            "version": {"asset_version_id": str(uuid7())},
            "replayed": False,
        })
        metadata = ManualUploadMetadata(display_name="Data Guard 运维手册")
        document = {
            "client_file_id": str(uuid7()),
            "file_name": "dg-guide.pdf",
            "display_name": "dg-guide.pdf",
            "media_type": "application/pdf",
            "body": b"pdf",
            "content_sha256": "a" * 64,
            "ordinal": 0,
            "role": "CONTENT",
            "required_flag": False,
        }

        receipt = await service.ingest_upload(
            domain_id=42,
            knowledge_base_id=uuid7(),
            context=SimpleNamespace(request_id="single-upload-test"),
            grouping_mode="EACH_FILE",
            bundle_title=metadata.display_name,
            documents=[document],
            metadata=metadata,
            idempotency_key="single-upload-test",
        )

        self.assertEqual("ACCEPTED", receipt["items"][0]["status"])
        forwarded = service.ingest_manual.await_args.kwargs
        self.assertEqual("Data Guard 运维手册", forwarded["metadata"].display_name)
        self.assertEqual("dg-guide.pdf", forwarded["file_name"])


class _KnowledgeUnitOfWork:
    def __init__(self, repository):
        self.operations_knowledge = repository
        self.commits = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def commit(self):
        self.commits += 1


class _SessionBoundEntity:
    def __init__(self, state, **values):
        self._state = state
        self._values = values

    def __getattr__(self, name):
        if not self._state.active:
            raise RuntimeError("detached entity attribute accessed")
        return self._values[name]


class _SessionState:
    active = True


class _DetachingKnowledgeUnitOfWork(_KnowledgeUnitOfWork):
    def __init__(self, repository, state):
        super().__init__(repository)
        self.state = state

    async def __aexit__(self, *_args):
        self.state.active = False
        return None


class OperationsKnowledgeAssistantBindingTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def service(repository, model_catalog=None, knowledge_core=None):
        return OperationsKnowledgeService(
            uow_factory=lambda: _KnowledgeUnitOfWork(repository),
            knowledge_core=knowledge_core or SimpleNamespace(),
            knowledge_assistant_id=uuid7(),
            knowledge_assistant_version_id=uuid7(),
            knowledge_assistant_instruction="仅依据运维知识证据回答",
            model_catalogs=(
                {1: model_catalog} if model_catalog is not None else {}
            ),
        )

    async def test_binding_is_configured_only_with_model_and_active_base(self):
        knowledge_base_id = uuid7()
        active_base = SimpleNamespace(
            knowledge_base_id=knowledge_base_id,
            domain_id=42,
            display_name="Oracle 运维库",
            description=None,
            collection_id=uuid7(),
            status="ACTIVE",
            row_version=1,
            created_by="operator",
            updated_by="operator",
            created_at=None,
            updated_at=None,
        )
        binding = SimpleNamespace(
            assistant_id=uuid7(),
            assistant_version_id=uuid7(),
            composer_model_id=None,
            row_version=1,
            updated_at=None,
        )
        repository = SimpleNamespace(
            get_assistant_binding=AsyncMock(return_value=binding),
            assistant_knowledge_base_ids=AsyncMock(
                return_value=[knowledge_base_id]
            ),
            get_knowledge_base=AsyncMock(return_value=active_base),
        )
        service = self.service(repository)

        missing_model = await service.get_knowledge_assistant_binding(
            domain_id=42
        )
        binding.composer_model_id = uuid7()
        configured = await service.get_knowledge_assistant_binding(
            domain_id=42
        )

        self.assertFalse(missing_model["configured"])
        self.assertTrue(configured["configured"])
        self.assertEqual(
            str(binding.composer_model_id),
            configured["composer_model_id"],
        )

    async def test_create_binding_persists_model_and_multiple_knowledge_bases(self):
        model_id = uuid7()
        knowledge_base_ids = (uuid7(), uuid7())
        repository = SimpleNamespace(
            get_knowledge_base=AsyncMock(side_effect=[
                SimpleNamespace(status="ACTIVE"),
                SimpleNamespace(status="ACTIVE"),
            ]),
            get_assistant_binding=AsyncMock(return_value=None),
            add_assistant_binding=AsyncMock(),
            replace_assistant_knowledge_bases=AsyncMock(),
        )
        catalog = SimpleNamespace(get_model=AsyncMock(return_value={
            "category": 1,
            "status": "ACTIVE",
        }))
        service = self.service(repository, catalog)
        service.get_knowledge_assistant_binding = AsyncMock(return_value={
            "configured": True,
        })
        context = SimpleNamespace(
            asserted_user_id="operator",
            client_id="portal",
        )

        result = await service.configure_knowledge_assistant_binding(
            domain_id=42,
            command=KnowledgeAssistantBindingWrite(
                composer_model_id=model_id,
                knowledge_base_ids=knowledge_base_ids,
            ),
            context=context,
        )

        self.assertTrue(result["configured"])
        catalog.get_model.assert_awaited_once_with(
            model_id, auth_context=context,
        )
        created = repository.add_assistant_binding.await_args.args[0]
        self.assertEqual(model_id, created.composer_model_id)
        repository.replace_assistant_knowledge_bases.assert_awaited_once_with(
            domain_id=42,
            knowledge_base_ids=knowledge_base_ids,
        )

    async def test_update_binding_rotates_version_and_model(self):
        old_version_id = uuid7()
        new_model_id = uuid7()
        knowledge_base_id = uuid7()
        binding = SimpleNamespace(
            assistant_id=uuid7(),
            assistant_version_id=old_version_id,
            composer_model_id=uuid7(),
            row_version=4,
            updated_by="old-user",
            updated_at=None,
        )
        repository = SimpleNamespace(
            get_knowledge_base=AsyncMock(return_value=SimpleNamespace(
                status="ACTIVE"
            )),
            get_assistant_binding=AsyncMock(return_value=binding),
            replace_assistant_knowledge_bases=AsyncMock(),
        )
        catalog = SimpleNamespace(get_model=AsyncMock(return_value={
            "category": 1,
            "status": "ACTIVE",
        }))
        service = self.service(repository, catalog)
        service.get_knowledge_assistant_binding = AsyncMock(return_value={
            "configured": True,
        })

        await service.configure_knowledge_assistant_binding(
            domain_id=42,
            command=KnowledgeAssistantBindingWrite(
                composer_model_id=new_model_id,
                knowledge_base_ids=(knowledge_base_id,),
                expected_row_version=4,
            ),
            context=SimpleNamespace(
                asserted_user_id="operator",
                client_id="portal",
            ),
        )

        self.assertEqual(new_model_id, binding.composer_model_id)
        self.assertNotEqual(old_version_id, binding.assistant_version_id)
        self.assertEqual(5, binding.row_version)

    async def test_binding_rejects_inactive_or_non_llm_model(self):
        for definition in (
            {"category": 1, "status": "DISABLED"},
            {"category": 2, "status": "ACTIVE"},
        ):
            with self.subTest(definition=definition):
                catalog = SimpleNamespace(
                    get_model=AsyncMock(return_value=definition)
                )
                service = self.service(SimpleNamespace(), catalog)
                with self.assertRaises(AIOpsApplicationError) as raised:
                    await service.configure_knowledge_assistant_binding(
                        domain_id=42,
                        command=KnowledgeAssistantBindingWrite(
                            composer_model_id=uuid7(),
                            knowledge_base_ids=(uuid7(),),
                        ),
                        context=SimpleNamespace(),
                    )
                self.assertEqual(409, raised.exception.status_code)
                self.assertEqual(
                    "AIOPS_KNOWLEDGE_ASSISTANT_MODEL_UNAVAILABLE",
                    raised.exception.code,
                )

    async def test_execution_spec_uses_persisted_binding_model_and_ids(self):
        assistant_id = uuid7()
        assistant_version_id = uuid7()
        composer_model_id = uuid7()
        knowledge_base_id = uuid7()
        collection_id = uuid7()
        revision_id = uuid7()
        asset_version_id = uuid7()
        state = _SessionState()
        repository = SimpleNamespace(
            get_assistant_binding=AsyncMock(return_value=SimpleNamespace(
                assistant_id=assistant_id,
                assistant_version_id=assistant_version_id,
                composer_model_id=composer_model_id,
            )),
            assistant_knowledge_base_ids=AsyncMock(
                return_value=[knowledge_base_id]
            ),
            list_published_candidates=AsyncMock(return_value=[(
                SimpleNamespace(),
                _SessionBoundEntity(
                    state,
                    asset_version_id=asset_version_id,
                ),
                _SessionBoundEntity(
                    state,
                    collection_id=collection_id,
                    bundle_revision_id=revision_id,
                ),
            )]),
        )
        knowledge_core = SimpleNamespace(bind_collection=AsyncMock())
        service = OperationsKnowledgeService(
            uow_factory=lambda: _DetachingKnowledgeUnitOfWork(
                repository, state
            ),
            knowledge_core=knowledge_core,
            knowledge_assistant_id=uuid7(),
            knowledge_assistant_version_id=uuid7(),
            knowledge_assistant_instruction="仅依据运维知识证据回答",
        )
        context = SimpleNamespace()

        spec = await service.knowledge_execution_spec(
            domain_id=42,
            max_security_level=2,
            business_context={},
            context=context,
        )

        self.assertEqual(str(assistant_id), spec["consumer_agent_id"])
        self.assertEqual(
            str(assistant_version_id),
            spec["consumer_agent_version_id"],
        )
        self.assertEqual(
            str(composer_model_id),
            spec["models"]["composer_llm"],
        )
        knowledge_core.bind_collection.assert_awaited_once_with(
            domain_id=42,
            agent_id=assistant_id,
            collection_id=collection_id,
            note="AIOps 内部运维知识助手检索授权",
            auth_context=context,
        )

    async def test_execution_spec_requires_persisted_model(self):
        repository = SimpleNamespace(
            get_assistant_binding=AsyncMock(return_value=SimpleNamespace(
                assistant_id=uuid7(),
                assistant_version_id=uuid7(),
                composer_model_id=None,
            )),
            assistant_knowledge_base_ids=AsyncMock(return_value=[uuid7()]),
        )
        service = self.service(repository)

        with self.assertRaises(AIOpsApplicationError) as raised:
            await service.knowledge_execution_spec(
                domain_id=42,
                max_security_level=2,
                business_context={},
                context=SimpleNamespace(),
            )

        self.assertEqual(422, raised.exception.status_code)
        self.assertEqual(
            "AIOPS_KNOWLEDGE_ASSISTANT_MODEL_REQUIRED",
            raised.exception.code,
        )


class OperationsKnowledgeLifecycleTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def service(repository, knowledge_core):
        return OperationsKnowledgeService(
            uow_factory=lambda: _KnowledgeUnitOfWork(repository),
            knowledge_core=knowledge_core,
            knowledge_assistant_id=uuid7(),
            knowledge_assistant_version_id=uuid7(),
            knowledge_assistant_instruction="",
        )

    async def test_collection_models_snapshot_id_before_transaction_closes(self):
        knowledge_base_id, collection_id = uuid7(), uuid7()
        state = _SessionState()
        repository = SimpleNamespace(get_knowledge_base=AsyncMock(
            return_value=_SessionBoundEntity(
                state, status="ACTIVE", collection_id=collection_id,
            )
        ))
        knowledge_core = SimpleNamespace(get_collection=AsyncMock(return_value={
            "collection_id": str(collection_id),
            "models": {"embedding": str(uuid7())},
        }))
        service = OperationsKnowledgeService(
            uow_factory=lambda: _DetachingKnowledgeUnitOfWork(repository, state),
            knowledge_core=knowledge_core,
            knowledge_assistant_id=uuid7(),
            knowledge_assistant_version_id=uuid7(),
            knowledge_assistant_instruction="",
        )
        context = SimpleNamespace()

        result = await service.get_collection_models(
            domain_id=42,
            knowledge_base_id=knowledge_base_id,
            context=context,
        )

        self.assertEqual(str(collection_id), result["collection_id"])
        knowledge_core.get_collection.assert_awaited_once_with(
            domain_id=42,
            collection_id=collection_id,
            auth_context=context,
        )

    async def test_pending_manual_can_be_deleted_after_kc_intake_is_rejected(self):
        asset_id, version_id = uuid7(), uuid7()
        collection_id, revision_id = uuid7(), uuid7()
        asset = SimpleNamespace(
            asset_id=asset_id,
            row_version=2,
            asset_kind="MANUAL",
        )
        version = SimpleNamespace(
            asset_version_id=version_id,
            status="REVIEW_REQUIRED",
        )
        index = SimpleNamespace(
            collection_id=collection_id,
            bundle_revision_id=revision_id,
            index_status="PENDING_APPROVAL",
        )
        repository = SimpleNamespace(
            get_asset=AsyncMock(return_value=asset),
            list_versions=AsyncMock(return_value=[version]),
            get_version_relations=AsyncMock(return_value={"indexes": [index]}),
            delete_asset=AsyncMock(),
        )
        knowledge_core = SimpleNamespace(
            review_user_intake=AsyncMock(),
            purge_revision_source_objects=AsyncMock(return_value={
                "purged_count": 1, "retained_count": 0,
            }),
        )
        service = self.service(repository, knowledge_core)

        await service.delete_asset(
            domain_id=42,
            asset_id=asset_id,
            expected_row_version=2,
            purge_source_objects=True,
            context=SimpleNamespace(),
        )

        knowledge_core.review_user_intake.assert_awaited_once_with(
            domain_id=42,
            collection_id=collection_id,
            bundle_revision_id=revision_id,
            decision="REJECT",
            comment="AIOps 用户删除误上传资料",
            auth_context=ANY,
        )
        knowledge_core.purge_revision_source_objects.assert_awaited_once_with(
            domain_id=42,
            collection_id=collection_id,
            bundle_revision_id=revision_id,
            auth_context=ANY,
        )
        repository.delete_asset.assert_awaited_once_with(
            asset=asset, versions=[version]
        )

    async def test_published_manual_cannot_be_deleted(self):
        asset_id, version_id = uuid7(), uuid7()
        asset = SimpleNamespace(
            asset_id=asset_id,
            row_version=1,
            asset_kind="MANUAL",
        )
        repository = SimpleNamespace(
            get_asset=AsyncMock(return_value=asset),
            list_versions=AsyncMock(return_value=[SimpleNamespace(
                asset_version_id=version_id,
                status="PUBLISHED",
            )]),
            get_version_relations=AsyncMock(),
            delete_asset=AsyncMock(),
        )
        knowledge_core = SimpleNamespace(review_user_intake=AsyncMock())
        service = self.service(repository, knowledge_core)

        with self.assertRaises(AIOpsApplicationError) as raised:
            await service.delete_asset(
                domain_id=42,
                asset_id=asset_id,
                expected_row_version=1,
                context=SimpleNamespace(),
            )

        self.assertEqual(409, raised.exception.status_code)
        knowledge_core.review_user_intake.assert_not_awaited()
        repository.delete_asset.assert_not_awaited()

    async def test_approve_manual_starts_tagging_without_calling_kc_or_llm(self):
        asset_id, version_id = uuid7(), uuid7()
        version = SimpleNamespace(
            asset_id=asset_id,
            status="REVIEW_REQUIRED",
            row_version=3,
        )
        asset = SimpleNamespace(
            current_version_id=None, status="REVIEW_REQUIRED", updated_by=None
        )
        index = SimpleNamespace(
            index_status="PENDING_APPROVAL", error_code=None, error_summary=None
        )
        repository = SimpleNamespace(
            get_version=AsyncMock(return_value=version),
            get_asset=AsyncMock(return_value=asset),
            add_review=AsyncMock(),
        )
        knowledge_core = SimpleNamespace(review_user_intake=AsyncMock())
        service = OperationsKnowledgeService(
            uow_factory=lambda: _KnowledgeUnitOfWork(repository),
            knowledge_core=knowledge_core,
            knowledge_assistant_id=uuid7(),
            knowledge_assistant_version_id=uuid7(),
            knowledge_assistant_instruction="",
        )
        detail = {
            "asset": {"asset_kind": "MANUAL"},
            "version": {"status": "REVIEW_REQUIRED", "row_version": 3},
            "indexes": [{
                "status": "PENDING_APPROVAL",
                "collection_id": str(uuid7()),
                "bundle_revision_id": str(uuid7()),
            }],
        }
        service.get_version = AsyncMock(return_value=detail)

        await service.approve_manual(
            domain_id=42,
            asset_version_id=version_id,
            expected_row_version=3,
            comment="资料已核对",
            context=SimpleNamespace(asserted_user_id="reviewer", client_id="portal"),
        )

        knowledge_core.review_user_intake.assert_not_awaited()
        self.assertEqual("TAGGING", version.status)
        self.assertEqual("TAGGING", asset.status)
        self.assertEqual(
            "APPROVED", version.tag_analysis_json["approval"]["status"]
        )
        self.assertEqual("PENDING_APPROVAL", index.index_status)
        self.assertEqual(
            "TAGGING", repository.add_review.await_args.args[0].after_status
        )

    async def test_tag_analysis_snapshots_entities_before_read_transaction_closes(self):
        asset_id, version_id, tag_id = uuid7(), uuid7(), uuid7()
        collection_id, bundle_id, revision_id = uuid7(), uuid7(), uuid7()
        document_version_id = uuid7()
        state = _SessionState()
        first_repository = SimpleNamespace(
            get_version=AsyncMock(return_value=_SessionBoundEntity(
                state,
                asset_id=asset_id,
                status="TAGGING",
                tag_analysis_json={"approval": {"status": "APPROVED"}},
            )),
            get_asset=AsyncMock(return_value=_SessionBoundEntity(
                state, asset_kind="MANUAL"
            )),
            get_version_relations=AsyncMock(return_value={
                "indexes": [_SessionBoundEntity(
                    state,
                    collection_id=collection_id,
                    bundle_id=bundle_id,
                    bundle_revision_id=revision_id,
                )]
            }),
            list_tags=AsyncMock(return_value=[_SessionBoundEntity(
                state,
                tag_id=tag_id,
                category="VENDOR",
                display_name="Oracle",
                description="Oracle 数据库资料",
            )]),
        )
        locked = SimpleNamespace(
            asset_id=asset_id, status="TAGGING", tag_analysis_json=None
        )
        locked_asset = SimpleNamespace(
            current_version_id=None, status="TAGGING"
        )
        second_repository = SimpleNamespace(
            get_version=AsyncMock(return_value=locked),
            get_asset=AsyncMock(return_value=locked_asset),
            get_version_relations=AsyncMock(return_value={
                "indexes": [SimpleNamespace(
                    index_status="PENDING_APPROVAL",
                    error_code=None,
                    error_summary=None,
                )]
            }),
            replace_tag_scopes=AsyncMock(),
        )
        units = iter([
            _DetachingKnowledgeUnitOfWork(first_repository, state),
            _KnowledgeUnitOfWork(second_repository),
        ])
        parser_llm_id = uuid7()
        knowledge_core = SimpleNamespace(
            get_collection=AsyncMock(return_value={
                "models": {"parser_llm": str(parser_llm_id)},
            }),
            extract_review_text=AsyncMock(return_value={
                "excerpts": [{
                    "document_version_id": str(document_version_id),
                    "file_name": "oracle-manual.pdf",
                    "content": "Oracle Exadata 运维手册",
                    "locator": {"page": 1},
                }],
                "warnings": [],
            }),
            review_user_intake=AsyncMock(),
        )
        structured_model = SimpleNamespace(generate_structured=AsyncMock(
            return_value=SimpleNamespace(
                output={
                    "schema_version": "OPS_KNOWLEDGE_TAG_ANALYSIS.v1",
                    "suggestions": [{
                        "tag_id": str(tag_id),
                        "confidence": 0.98,
                        "evidence": [{
                            "document_version_id": str(document_version_id),
                            "file_name": "oracle-manual.pdf",
                            "quote": "Oracle Exadata",
                            "locator": {"page": 1},
                        }],
                    }],
                },
                receipt=SimpleNamespace(model_dump=lambda **_kwargs: {
                    "model_call_id": str(uuid7())
                }),
            )
        ))
        prompt = SimpleNamespace(
            content="从受控词表选择标签",
            ref=lambda: {"prompt_id": "knowledge_tag_classify"},
        )
        service = OperationsKnowledgeService(
            uow_factory=lambda: next(units),
            knowledge_core=knowledge_core,
            knowledge_assistant_id=uuid7(),
            knowledge_assistant_version_id=uuid7(),
            knowledge_assistant_instruction="",
            model_catalogs={1: SimpleNamespace(get_model=AsyncMock(
                return_value={"served_model_name": "tag-model"}
            ))},
            structured_model=structured_model,
            prompt_registry=SimpleNamespace(resolve=AsyncMock(return_value=prompt)),
        )
        service.get_version = AsyncMock(return_value={"status": "PROCESSING"})

        result = await service.analyze_tags(
            domain_id=42,
            asset_version_id=version_id,
            context=SimpleNamespace(),
        )

        self.assertEqual({"status": "PROCESSING"}, result)
        self.assertEqual("PROCESSING", locked.status, locked.tag_analysis_json)
        self.assertEqual("PROCESSING", locked_asset.status)
        self.assertEqual("Oracle", locked.tag_analysis_json["suggestions"][0]["display_name"])
        knowledge_core.extract_review_text.assert_awaited_once()
        knowledge_core.review_user_intake.assert_awaited_once()
        tag_snapshot = second_repository.replace_tag_scopes.await_args.kwargs[
            "tags"
        ][0]
        self.assertEqual(tag_id, tag_snapshot.tag_id)
        self.assertEqual("VENDOR", tag_snapshot.category)
        self.assertEqual("Oracle", tag_snapshot.display_name)
        structured_model.generate_structured.assert_awaited_once()

    async def test_unapproved_legacy_tagging_state_returns_to_review(self):
        version_id, asset_id = uuid7(), uuid7()
        version = SimpleNamespace(
            asset_id=asset_id, status="TAGGING", tag_analysis_json=None
        )
        asset = SimpleNamespace(
            asset_kind="MANUAL", current_version_id=None, status="TAGGING"
        )
        repository = SimpleNamespace(
            get_version=AsyncMock(return_value=version),
            get_asset=AsyncMock(return_value=asset),
            get_version_relations=AsyncMock(return_value={"indexes": []}),
            list_tags=AsyncMock(return_value=[]),
        )
        structured_model = SimpleNamespace(generate_structured=AsyncMock())
        service = OperationsKnowledgeService(
            uow_factory=lambda: _KnowledgeUnitOfWork(repository),
            knowledge_core=SimpleNamespace(),
            knowledge_assistant_id=uuid7(),
            knowledge_assistant_version_id=uuid7(),
            knowledge_assistant_instruction="",
            model_catalogs={1: SimpleNamespace()},
            structured_model=structured_model,
            prompt_registry=SimpleNamespace(),
        )
        service.get_version = AsyncMock(return_value={
            "version": {"status": "REVIEW_REQUIRED"}
        })

        await service.analyze_tags(
            domain_id=1,
            asset_version_id=version_id,
            context=SimpleNamespace(),
        )

        self.assertEqual("REVIEW_REQUIRED", version.status)
        self.assertEqual("REVIEW_REQUIRED", asset.status)
        structured_model.generate_structured.assert_not_awaited()

    async def test_tag_analysis_retries_when_knowledge_core_is_temporarily_unavailable(self):
        asset_id, version_id = uuid7(), uuid7()
        version = SimpleNamespace(
            asset_id=asset_id,
            status="TAGGING",
            tag_analysis_json={"approval": {"status": "APPROVED"}},
        )
        repository = SimpleNamespace(
            get_version=AsyncMock(return_value=version),
            get_asset=AsyncMock(return_value=SimpleNamespace(
                asset_kind="MANUAL"
            )),
            get_version_relations=AsyncMock(return_value={
                "indexes": [SimpleNamespace(
                    collection_id=uuid7(),
                    bundle_id=uuid7(),
                    bundle_revision_id=uuid7(),
                )]
            }),
            list_tags=AsyncMock(return_value=[SimpleNamespace(
                tag_id=uuid7(),
                category="VENDOR",
                display_name="Oracle",
                description=None,
            )]),
        )
        knowledge_core = SimpleNamespace(extract_review_text=AsyncMock(
            side_effect=KnowledgeCoreClientError(
                status_code=503,
                code="KNOWLEDGE_CORE_UNAVAILABLE",
                message="Knowledge Core 暂时不可用",
            )
        ))
        service = OperationsKnowledgeService(
            uow_factory=lambda: _KnowledgeUnitOfWork(repository),
            knowledge_core=knowledge_core,
            knowledge_assistant_id=uuid7(),
            knowledge_assistant_version_id=uuid7(),
            knowledge_assistant_instruction="",
            model_catalogs={1: SimpleNamespace()},
            structured_model=SimpleNamespace(),
            prompt_registry=SimpleNamespace(),
        )

        with self.assertRaises(AIOpsApplicationError) as raised:
            await service.analyze_tags(
                domain_id=42,
                asset_version_id=version_id,
                context=SimpleNamespace(),
            )

        self.assertEqual("AIOPS_KNOWLEDGE_KC_UNAVAILABLE", raised.exception.code)
        self.assertEqual(503, raised.exception.status_code)
        self.assertTrue(raised.exception.retryable)
        self.assertEqual("TAGGING", version.status)

    async def test_tag_analysis_preserves_preview_warning_when_text_is_empty(self):
        asset_id, version_id = uuid7(), uuid7()
        version = SimpleNamespace(
            asset_id=asset_id,
            status="TAGGING",
            tag_analysis_json={"approval": {"status": "APPROVED"}},
        )
        asset = SimpleNamespace(
            asset_kind="MANUAL", current_version_id=None, status="TAGGING"
        )
        repository = SimpleNamespace(
            get_version=AsyncMock(return_value=version),
            get_asset=AsyncMock(return_value=asset),
            get_version_relations=AsyncMock(return_value={
                "indexes": [SimpleNamespace(
                    collection_id=uuid7(),
                    bundle_id=uuid7(),
                    bundle_revision_id=uuid7(),
                )]
            }),
            list_tags=AsyncMock(return_value=[SimpleNamespace(
                tag_id=uuid7(),
                category="VENDOR",
                display_name="Oracle",
                description=None,
            )]),
        )
        warning = "Oracle Exadata 手册 审批预分析失败：普通解析RuntimeError；OCR未抽取到文字"
        knowledge_core = SimpleNamespace(extract_review_text=AsyncMock(
            return_value={"excerpts": [], "warnings": [warning]}
        ))
        service = OperationsKnowledgeService(
            uow_factory=lambda: _KnowledgeUnitOfWork(repository),
            knowledge_core=knowledge_core,
            knowledge_assistant_id=uuid7(),
            knowledge_assistant_version_id=uuid7(),
            knowledge_assistant_instruction="",
            model_catalogs={1: SimpleNamespace()},
            structured_model=SimpleNamespace(),
            prompt_registry=SimpleNamespace(),
        )
        service.get_version = AsyncMock(return_value={
            "version": {"status": "TAGGING_FAILED"}
        })

        await service.analyze_tags(
            domain_id=42,
            asset_version_id=version_id,
            context=SimpleNamespace(),
        )

        self.assertEqual("TAGGING_FAILED", version.status)
        self.assertEqual("TAGGING_FAILED", asset.status)
        self.assertEqual(
            warning, version.tag_analysis_json["error_summary"]
        )

    async def test_ready_manual_is_published_automatically(self):
        asset_id, version_id = uuid7(), uuid7()
        bundle_id, revision_id, collection_id = uuid7(), uuid7(), uuid7()
        version = SimpleNamespace(
            asset_id=asset_id,
            asset_version_id=version_id,
            status="PROCESSING",
            profile_json={"schema_version": "OPS_MANUAL_PROFILE.v1", "warnings": []},
            extraction_warnings_json=[],
            published_at=None,
        )
        asset = SimpleNamespace(
            asset_kind="MANUAL", current_version_id=None, status="PROCESSING"
        )
        index = SimpleNamespace(
            bundle_revision_id=revision_id,
            index_status="PROCESSING",
            last_checked_at=None,
            expected_row_version=None,
            error_code=None,
            error_summary=None,
        )
        repository = SimpleNamespace(
            get_version=AsyncMock(return_value=version),
            get_asset=AsyncMock(return_value=asset),
            get_version_relations=AsyncMock(return_value={
                "indexes": [index], "scopes": [], "sources": []
            }),
        )
        knowledge_core = SimpleNamespace(
            get_bundle_status=AsyncMock(return_value={
                "current_revision_id": str(revision_id),
                "availability_status": "READY",
                "row_version": 9,
            }),
            get_revision_status=AsyncMock(return_value={
                "status": "READY",
                "source_revision": f"{version_id}:profile-v1",
                "members": [{
                    "document_role": "CONTENT",
                    "document_version_id": str(uuid7()),
                    "member_status": "READY",
                }],
            }),
        )
        service = self.service(repository, knowledge_core)
        detail = {
            "asset": {
                "asset_id": str(asset_id), "asset_kind": "MANUAL",
                "display_name": "Oracle 手册", "security_level": 1,
            },
            "version": {
                "status": "PROCESSING", "profile": version.profile_json,
                "source_hash": "a" * 64,
            },
            "indexes": [{
                "collection_id": str(collection_id),
                "bundle_id": str(bundle_id),
                "bundle_revision_id": str(revision_id),
            }],
        }
        service.get_version = AsyncMock(side_effect=[detail, {"published": True}])

        result = await service.reconcile_version(
            domain_id=42,
            asset_version_id=version_id,
            context=SimpleNamespace(),
        )

        self.assertEqual({"published": True}, result)
        self.assertEqual("PUBLISHED", version.status)
        self.assertEqual("PUBLISHED", asset.status)
        self.assertEqual(version_id, asset.current_version_id)
        self.assertEqual("READY", index.index_status)

    async def test_reprocessing_revision_ignores_previous_publication_failure(self):
        asset_id, version_id = uuid7(), uuid7()
        bundle_id, revision_id, collection_id = uuid7(), uuid7(), uuid7()
        version = SimpleNamespace(
            asset_id=asset_id,
            asset_version_id=version_id,
            status="FAILED",
            profile_json=None,
            extraction_warnings_json=[],
        )
        asset = SimpleNamespace(
            asset_kind="MANUAL", current_version_id=None, status="FAILED"
        )
        index = SimpleNamespace(
            bundle_revision_id=revision_id,
            index_status="FAILED",
            last_checked_at=None,
            expected_row_version=None,
            error_code="PARSER_MODEL_NOT_FOUND",
            error_summary="上一轮解析失败",
        )
        repository = SimpleNamespace(
            get_version=AsyncMock(return_value=version),
            get_asset=AsyncMock(return_value=asset),
            get_version_relations=AsyncMock(return_value={
                "indexes": [index], "scopes": [], "sources": []
            }),
        )
        knowledge_core = SimpleNamespace(
            get_bundle_status=AsyncMock(return_value={
                "current_revision_id": str(revision_id),
                "availability_status": "EMPTY",
                "row_version": 10,
            }),
            get_revision_status=AsyncMock(return_value={
                "status": "PROCESSING",
                "publication_status": "FAILED",
                "publication_failure_code": "PREVIOUS_INDEX_FAILED",
                "publication_failure_message": "上一轮索引失败",
                "members": [{
                    "document_role": "CONTENT",
                    "document_version_id": str(uuid7()),
                    "member_status": "RECEIVED",
                }],
            }),
        )
        service = self.service(repository, knowledge_core)
        detail = {
            "asset": {
                "asset_id": str(asset_id), "asset_kind": "MANUAL",
                "display_name": "Oracle Exadata 手册", "security_level": 1,
            },
            "version": {
                "status": "FAILED", "profile": None,
                "source_hash": "a" * 64,
            },
            "indexes": [{
                "collection_id": str(collection_id),
                "bundle_id": str(bundle_id),
                "bundle_revision_id": str(revision_id),
            }],
        }
        service.get_version = AsyncMock(
            side_effect=[detail, {"version": {"status": "PROCESSING"}}]
        )

        result = await service.reconcile_version(
            domain_id=42,
            asset_version_id=version_id,
            context=SimpleNamespace(),
        )

        self.assertEqual({"version": {"status": "PROCESSING"}}, result)
        self.assertEqual("PROCESSING", version.status)
        self.assertEqual("PROCESSING", asset.status)
        self.assertEqual("PROCESSING", index.index_status)
        self.assertIsNone(index.error_code)
        self.assertIsNone(index.error_summary)

    async def test_profile_revision_preserves_every_manual_source_member(self):
        asset_id, version_id = uuid7(), uuid7()
        bundle_id, current_revision_id, stale_revision_id = (
            uuid7(), uuid7(), uuid7()
        )
        replacement_bundle_id, replacement_revision_id = uuid7(), uuid7()
        collection_id = uuid7()
        document_ids = (uuid7(), uuid7())
        bodies = (b"primary manual", b"supporting appendix")
        digests = tuple(hashlib.sha256(body).hexdigest() for body in bodies)
        profile = {"schema_version": "OPS_MANUAL_PROFILE.v1", "warnings": []}
        version = SimpleNamespace(
            asset_id=asset_id,
            asset_version_id=version_id,
            status="PROCESSING",
            profile_json=profile,
            extraction_warnings_json=[],
        )
        asset = SimpleNamespace(
            asset_kind="MANUAL", current_version_id=None, status="PROCESSING"
        )
        index = SimpleNamespace(
            bundle_id=bundle_id,
            bundle_revision_id=stale_revision_id,
            index_status="PROCESSING",
            last_checked_at=None,
            expected_row_version=None,
            error_code=None,
            error_summary=None,
        )
        repository = SimpleNamespace(
            get_version=AsyncMock(return_value=version),
            get_asset=AsyncMock(return_value=asset),
            get_version_relations=AsyncMock(return_value={
                "indexes": [index], "scopes": [], "sources": []
            }),
        )

        def stream_source_file(**kwargs):
            position = document_ids.index(kwargs["document_version_id"])
            return SimpleNamespace(
                headers={"content-type": "application/pdf"},
                body=_body_chunks(bodies[position]),
            )

        knowledge_core = SimpleNamespace(
            get_bundle_status=AsyncMock(return_value={
                "source_id": str(version_id),
                "current_revision_id": str(current_revision_id),
                "availability_status": "READY",
                "row_version": 12,
            }),
            get_revision_status=AsyncMock(side_effect=[
                KnowledgeCoreClientError(
                    status_code=404,
                    code="KNOWLEDGE_REVISION_NOT_FOUND",
                    message="Revision not found",
                ),
                {
                    "status": "READY",
                    "source_revision": str(version_id),
                    "members": [
                        {
                            "document_role": "CONTENT",
                            "document_version_id": str(document_ids[0]),
                            "declared_name": "exadata.pdf",
                            "declared_mime_type": "application/pdf",
                            "ordinal": 0,
                            "required_flag": True,
                            "member_status": "READY",
                        },
                        {
                            "document_role": "SUPPLEMENT",
                            "document_version_id": str(document_ids[1]),
                            "declared_name": "appendix.pdf",
                            "declared_mime_type": "application/pdf",
                            "ordinal": 1,
                            "required_flag": False,
                            "member_status": "READY",
                        },
                    ],
                },
            ]),
            stream_source_file=AsyncMock(side_effect=stream_source_file),
            ingest_user_bundle=AsyncMock(return_value=SimpleNamespace(
                payload={"items": [{
                    "status": "ACCEPTED",
                    "bundle_id": str(replacement_bundle_id),
                    "bundle_revision_id": str(replacement_revision_id),
                }]}
            )),
            review_user_intake=AsyncMock(),
        )
        service = self.service(repository, knowledge_core)
        detail = {
            "asset": {
                "asset_id": str(asset_id), "asset_kind": "MANUAL",
                "display_name": "Oracle Exadata 手册", "security_level": 1,
            },
            "version": {
                "status": "PROCESSING", "profile": profile,
                "source_hash": "bundle-digest",
            },
            "sources": [
                {
                    "source_kind": "MANUAL_FILE",
                    "external_id": "exadata.pdf",
                    "content_hash": digests[0],
                    "locator": {
                        "ordinal": 0, "role": "CONTENT",
                        "required_flag": True,
                    },
                },
                {
                    "source_kind": "MANUAL_FILE",
                    "external_id": "appendix.pdf",
                    "content_hash": digests[1],
                    "locator": {
                        "ordinal": 1, "role": "SUPPLEMENT",
                        "required_flag": False,
                    },
                },
            ],
            "indexes": [{
                "collection_id": str(collection_id),
                "bundle_id": str(bundle_id),
                "bundle_revision_id": str(stale_revision_id),
            }],
        }
        service.get_version = AsyncMock(
            side_effect=[detail, {"version": {"status": "PROCESSING"}}]
        )

        await service.reconcile_version(
            domain_id=42,
            asset_version_id=version_id,
            context=SimpleNamespace(),
        )

        documents = knowledge_core.ingest_user_bundle.await_args.kwargs[
            "documents"
        ]
        self.assertEqual(3, len(documents))
        self.assertEqual(
            ["exadata.pdf", "appendix.pdf"],
            [item["file_name"] for item in documents[:2]],
        )
        self.assertEqual(
            ["CONTENT", "SUPPLEMENT"],
            [item["role"] for item in documents[:2]],
        )
        self.assertEqual([True, False], [
            item["required_flag"] for item in documents[:2]
        ])
        self.assertEqual(list(digests), [
            item["content_sha256"] for item in documents[:2]
        ])
        self.assertEqual(
            str(version_id),
            knowledge_core.ingest_user_bundle.await_args.kwargs[
                "client_bundle_id"
            ],
        )
        self.assertEqual(replacement_bundle_id, index.bundle_id)
        self.assertEqual(replacement_revision_id, index.bundle_revision_id)
        self.assertEqual(
            current_revision_id,
            knowledge_core.stream_source_file.await_args_list[0].kwargs[
                "bundle_revision_id"
            ],
        )

    async def test_explicit_model_configuration_updates_selected_knowledge_base(self):
        knowledge_base_id = uuid7()
        collection_id = uuid7()
        repository = SimpleNamespace(get_knowledge_base=AsyncMock(
            return_value=SimpleNamespace(
                knowledge_base_id=knowledge_base_id,
                collection_id=collection_id,
                status="ACTIVE",
            )
        ))
        knowledge_core = SimpleNamespace(
            get_collection=AsyncMock(return_value={
                "collection_id": str(collection_id),
                "row_version": 3,
            }),
            update_collection_models=AsyncMock(side_effect=lambda **kwargs: kwargs["payload"]),
        )
        models = {
            "parser_ocr": uuid7(),
            "parser_llm": uuid7(),
            "retrieval_llm": uuid7(),
            "embedding": uuid7(),
        }
        categories = {
            models["parser_ocr"]: 6,
            models["parser_llm"]: 1,
            models["retrieval_llm"]: 1,
            models["embedding"]: 2,
        }
        model_catalogs = {
            category: SimpleNamespace(get_model=AsyncMock(
                side_effect=lambda model_id, **_kwargs: {
                    "category": categories[model_id], "status": "ACTIVE",
                }
            ))
            for category in set(categories.values())
        }
        service = OperationsKnowledgeService(
            uow_factory=lambda: _KnowledgeUnitOfWork(repository),
            knowledge_core=knowledge_core,
            knowledge_assistant_id=uuid7(),
            knowledge_assistant_version_id=uuid7(),
            knowledge_assistant_instruction="",
            model_catalogs=model_catalogs,
        )

        context = SimpleNamespace()
        result = await service.configure_collections(
            domain_id=42,
            knowledge_base_id=knowledge_base_id,
            models=models,
            context=context,
        )

        self.assertEqual(str(knowledge_base_id), result["knowledge_base_id"])
        knowledge_core.update_collection_models.assert_awaited_once()
        call = knowledge_core.update_collection_models.await_args
        self.assertEqual(
            {role: str(model_id) for role, model_id in models.items()},
            call.kwargs["payload"]["models"],
        )
        self.assertEqual(3, call.kwargs["payload"]["expected_row_version"])
        model_catalogs[6].get_model.assert_awaited_once_with(
            models["parser_ocr"], auth_context=context,
        )
        self.assertEqual(2, model_catalogs[1].get_model.await_count)
        model_catalogs[2].get_model.assert_awaited_once()

    async def test_model_configuration_preserves_kc_business_error(self):
        knowledge_base_id = uuid7()
        collection_id = uuid7()
        repository = SimpleNamespace(get_knowledge_base=AsyncMock(
            return_value=SimpleNamespace(
                knowledge_base_id=knowledge_base_id,
                collection_id=collection_id,
                status="ACTIVE",
            )
        ))
        knowledge_core = SimpleNamespace(
            get_collection=AsyncMock(return_value={"row_version": 1}),
            update_collection_models=AsyncMock(side_effect=KnowledgeCoreClientError(
                status_code=422,
                code="INVALID_COLLECTION_MODELS",
                message="模型角色绑定无效",
            )),
        )
        models = {
            "parser_ocr": uuid7(),
            "parser_llm": uuid7(),
            "retrieval_llm": uuid7(),
            "embedding": uuid7(),
        }
        categories = {
            models["parser_ocr"]: 6,
            models["parser_llm"]: 1,
            models["retrieval_llm"]: 1,
            models["embedding"]: 2,
        }
        service = OperationsKnowledgeService(
            uow_factory=lambda: _KnowledgeUnitOfWork(repository),
            knowledge_core=knowledge_core,
            knowledge_assistant_id=uuid7(),
            knowledge_assistant_version_id=uuid7(),
            knowledge_assistant_instruction="",
            model_catalogs={category: SimpleNamespace(get_model=AsyncMock(
                side_effect=lambda model_id, **_kwargs: {
                    "category": categories[model_id], "status": "ACTIVE",
                }
            )) for category in set(categories.values())},
        )

        with self.assertRaises(AIOpsApplicationError) as raised:
            await service.configure_collections(
                domain_id=42,
                knowledge_base_id=knowledge_base_id,
                models=models,
                context=SimpleNamespace(),
            )

        self.assertEqual("INVALID_COLLECTION_MODELS", raised.exception.code)
        self.assertEqual(422, raised.exception.status_code)
        self.assertFalse(raised.exception.retryable)
        self.assertEqual("模型角色绑定无效", raised.exception.message)

    async def test_editing_knowledge_base_updates_name_and_models_together(self):
        knowledge_base_id, collection_id = uuid7(), uuid7()
        row = SimpleNamespace(
            knowledge_base_id=knowledge_base_id,
            collection_id=collection_id,
            status="ACTIVE",
            row_version=3,
            display_name="旧名称",
            description=None,
            updated_by=None,
            updated_at=None,
        )
        repository = SimpleNamespace(
            get_knowledge_base=AsyncMock(return_value=row),
            find_knowledge_base_name=AsyncMock(return_value=None),
        )
        knowledge_core = SimpleNamespace(
            get_collection=AsyncMock(return_value={"row_version": 8}),
            update_collection_models=AsyncMock(return_value={"row_version": 9}),
        )
        models = {
            "parser_ocr": uuid7(),
            "parser_llm": uuid7(),
            "retrieval_llm": uuid7(),
            "embedding": uuid7(),
        }
        categories = {
            models["parser_ocr"]: 6,
            models["parser_llm"]: 1,
            models["retrieval_llm"]: 1,
            models["embedding"]: 2,
        }
        service = OperationsKnowledgeService(
            uow_factory=lambda: _KnowledgeUnitOfWork(repository),
            knowledge_core=knowledge_core,
            knowledge_assistant_id=uuid7(),
            knowledge_assistant_version_id=uuid7(),
            knowledge_assistant_instruction="",
            model_catalogs={category: SimpleNamespace(get_model=AsyncMock(
                side_effect=lambda model_id, **_kwargs: {
                    "category": categories[model_id], "status": "ACTIVE",
                }
            )) for category in set(categories.values())},
        )
        service.get_knowledge_base = AsyncMock(return_value={
            "knowledge_base_id": str(knowledge_base_id)
        })

        await service.update_knowledge_base(
            domain_id=42,
            knowledge_base_id=knowledge_base_id,
            command=KnowledgeBaseWrite(
                display_name="新名称",
                description="新说明",
                models=models,
                expected_row_version=3,
            ),
            context=SimpleNamespace(asserted_user_id="operator", client_id="portal"),
        )

        self.assertEqual("新名称", row.display_name)
        self.assertEqual("新说明", row.description)
        self.assertEqual(4, row.row_version)
        knowledge_core.update_collection_models.assert_awaited_once()
        self.assertEqual(
            {role: str(model_id) for role, model_id in models.items()},
            knowledge_core.update_collection_models.await_args.kwargs["payload"]["models"],
        )

if __name__ == "__main__":
    unittest.main()
