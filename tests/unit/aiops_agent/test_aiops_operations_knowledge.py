"""运维知识提炼合同与原文命令边界测试。"""

from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from pydantic import ValidationError

from aiops_agent.application.operations_knowledge import (
    OperationsKnowledgeService,
    _bundle_source_hash,
)
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.contracts.knowledge import KnowledgeBaseWrite, ManualUploadMetadata
from aiops_agent.repositories.knowledge import OperationsKnowledgeRepository
from platform_core.identity import uuid7
from platform_clients.knowledge_core import KnowledgeCoreClientError


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
            knowledge_assistant_models={},
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


class OperationsKnowledgeLifecycleTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def service(repository, knowledge_core):
        return OperationsKnowledgeService(
            uow_factory=lambda: _KnowledgeUnitOfWork(repository),
            knowledge_core=knowledge_core,
            knowledge_assistant_id=uuid7(),
            knowledge_assistant_version_id=uuid7(),
            knowledge_assistant_models={},
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
            knowledge_assistant_models={},
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

    async def test_approve_manual_freezes_selected_tags_and_starts_kc_parse(self):
        asset_id, version_id, tag_id = uuid7(), uuid7(), uuid7()
        version = SimpleNamespace(
            asset_id=asset_id,
            status="REVIEW_REQUIRED",
            row_version=3,
            tag_analysis_json={"status": "READY", "suggestions": []},
        )
        asset = SimpleNamespace(
            current_version_id=None, status="REVIEW_REQUIRED", updated_by=None
        )
        index = SimpleNamespace(
            index_status="PENDING_APPROVAL", error_code=None, error_summary=None
        )
        tag = SimpleNamespace(
            tag_id=tag_id, category="VENDOR", display_name="Oracle"
        )
        repository = SimpleNamespace(
            list_tags=AsyncMock(return_value=[tag]),
            get_version=AsyncMock(return_value=version),
            get_asset=AsyncMock(return_value=asset),
            get_version_relations=AsyncMock(return_value={"indexes": [index]}),
            replace_tag_scopes=AsyncMock(),
            add_review=AsyncMock(),
        )
        knowledge_core = SimpleNamespace(review_user_intake=AsyncMock())
        service = self.service(repository, knowledge_core)
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
            tag_ids=(str(tag_id),),
            comment="证据已核对",
            context=SimpleNamespace(asserted_user_id="reviewer", client_id="portal"),
        )

        knowledge_core.review_user_intake.assert_awaited_once()
        repository.replace_tag_scopes.assert_awaited_once_with(
            version=version, tags=[tag]
        )
        self.assertEqual("PROCESSING", version.status)
        self.assertEqual("PROCESSING", index.index_status)
        self.assertEqual([str(tag_id)], version.tag_analysis_json["approved_tag_ids"])

    async def test_tag_analysis_snapshots_entities_before_read_transaction_closes(self):
        asset_id, version_id, tag_id = uuid7(), uuid7(), uuid7()
        collection_id, bundle_id, revision_id = uuid7(), uuid7(), uuid7()
        document_version_id = uuid7()
        state = _SessionState()
        first_repository = SimpleNamespace(
            get_version=AsyncMock(return_value=_SessionBoundEntity(
                state, asset_id=asset_id, status="TAGGING"
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
            knowledge_assistant_models={"composer_llm": uuid7()},
            knowledge_assistant_instruction="",
            model_catalog=SimpleNamespace(get_model=AsyncMock(
                return_value={"served_model_name": "tag-model"}
            )),
            structured_model=structured_model,
            prompt_registry=SimpleNamespace(resolve=AsyncMock(return_value=prompt)),
        )
        service.get_version = AsyncMock(return_value={"status": "REVIEW_REQUIRED"})

        result = await service.analyze_tags(
            domain_id=42,
            asset_version_id=version_id,
            context=SimpleNamespace(),
        )

        self.assertEqual({"status": "REVIEW_REQUIRED"}, result)
        self.assertEqual("REVIEW_REQUIRED", locked.status, locked.tag_analysis_json)
        self.assertEqual("REVIEW_REQUIRED", locked_asset.status)
        self.assertEqual("Oracle", locked.tag_analysis_json["suggestions"][0]["display_name"])
        knowledge_core.extract_review_text.assert_awaited_once()
        structured_model.generate_structured.assert_awaited_once()

    async def test_tag_analysis_retries_when_knowledge_core_is_temporarily_unavailable(self):
        asset_id, version_id = uuid7(), uuid7()
        version = SimpleNamespace(asset_id=asset_id, status="TAGGING")
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
            knowledge_assistant_models={"composer_llm": uuid7()},
            knowledge_assistant_instruction="",
            model_catalog=SimpleNamespace(),
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
        model_catalog = SimpleNamespace(get_model=AsyncMock(
            side_effect=lambda model_id, **_kwargs: {
                "category": categories[model_id], "status": "ACTIVE",
            }
        ))
        service = OperationsKnowledgeService(
            uow_factory=lambda: _KnowledgeUnitOfWork(repository),
            knowledge_core=knowledge_core,
            knowledge_assistant_id=uuid7(),
            knowledge_assistant_version_id=uuid7(),
            knowledge_assistant_models={"composer_llm": uuid7()},
            knowledge_assistant_instruction="",
            model_catalog=model_catalog,
        )

        result = await service.configure_collections(
            domain_id=42,
            knowledge_base_id=knowledge_base_id,
            models=models,
            context=SimpleNamespace(),
        )

        self.assertEqual(str(knowledge_base_id), result["knowledge_base_id"])
        knowledge_core.update_collection_models.assert_awaited_once()
        call = knowledge_core.update_collection_models.await_args
        self.assertEqual(
            {role: str(model_id) for role, model_id in models.items()},
            call.kwargs["payload"]["models"],
        )

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
            knowledge_assistant_models={},
            knowledge_assistant_instruction="",
            model_catalog=SimpleNamespace(get_model=AsyncMock(
                side_effect=lambda model_id, **_kwargs: {
                    "category": categories[model_id], "status": "ACTIVE",
                }
            )),
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
