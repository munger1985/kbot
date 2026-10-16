"""运维知识提炼合同与原文命令边界测试。"""

from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from pydantic import ValidationError

from aiops_agent.application.operations_knowledge import (
    OperationsKnowledgeService,
    _bundle_source_hash,
)
from aiops_agent.contracts.knowledge import ManualUploadMetadata
from aiops_agent.repositories.knowledge import OperationsKnowledgeRepository
from platform_core.identity import uuid7


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
    def test_upload_metadata_rejects_internal_collection_fields(self):
        with self.assertRaises(ValidationError):
            ManualUploadMetadata.model_validate({
                "display_name": "Oracle 手册",
                "collection_id": str(uuid7()),
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

    def test_manual_profile_marks_declared_database_conflict_for_review(self):
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
        self.assertEqual(1, len(profile.warnings))
        self.assertIn("用户=ORACLE", profile.warnings[0])
        self.assertIn("原文=POSTGRESQL", profile.warnings[0])

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

if __name__ == "__main__":
    unittest.main()
