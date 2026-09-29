"""数据库实施 Runbook PDF 下载边界测试。"""

from __future__ import annotations

import asyncio
import io
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
import zipfile

from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.implementation.pdf import (
    _RunbookCodeBlock,
    _formatted_code,
    _wrapped_code,
    render_implementation_runbook_pdf,
)
from aiops_agent.application.implementation.markdown import (
    render_implementation_runbook_markdown,
)
from aiops_agent.application.implementation import (
    compile_implementation_runbook,
)
from aiops_agent.application.implementation.artifacts import (
    generated_artifact_payloads,
)
from aiops_agent.application.turns import ConversationTurnService
from platform_core.contracts.aiops import ImplementationProfile
from platform_core.identity import uuid7


def _context(uow):
    context = AsyncMock()
    context.__aenter__.return_value = uow
    context.__aexit__.return_value = None
    return context


def _historical_payload() -> dict:
    return {
        "schema_version": "AIOPS_IMPLEMENTATION_RUNBOOK.v1",
        "profile": "ORACLE_ADG_BUILD",
        "title": "历史 ADG 实施文档",
        "status": "READY",
        "execution_policy": "仅生成文档，不执行命令。",
        "phases": [
            {
                "phase_id": "primary",
                "title": "主库整改",
                "objective": "补齐当前未满足项。",
                "steps": [
                    {
                        "step_id": "primary.archivelog",
                        "title": "启用归档",
                        "applicability": "REQUIRED",
                        "rationale": "当前为 NOARCHIVELOG。",
                        "commands": [
                            {
                                "command_type": "SQLPLUS",
                                "title": "启用归档",
                                "content": "ALTER DATABASE ARCHIVELOG;",
                            }
                        ],
                    }
                ],
            }
        ],
    }


class ImplementationRunbookPdfTest(unittest.TestCase):
    def test_pdf_code_block_splits_without_losing_lines(self) -> None:
        source_lines = [f"SELECT {index} FROM dual;" for index in range(20)]
        block = _RunbookCodeBlock(
            title="SQLPLUS · 分页验证",
            code="\n".join(source_lines),
            text_font_name="Helvetica",
        )

        parts = block.split(420, 80)

        self.assertEqual(2, len(parts))
        self.assertEqual(
            source_lines,
            [line for part in parts for line in part.lines],
        )
        self.assertEqual("SQLPLUS · 分页验证", parts[0].title)
        self.assertEqual("SQLPLUS · 分页验证（续）", parts[1].title)

    def test_markdown_projection_contains_toc_and_original_command(self) -> None:
        payload = _historical_payload()
        payload["phases"][0]["steps"][0]["commands"][0]["content"] = (
            "ALTER SESSION SET CONTAINER=CDB$ROOT;\n"
            "ALTER DATABASE ARCHIVELOG;"
        )

        content = render_implementation_runbook_markdown(payload)

        self.assertIn("# 历史 ADG 实施文档", content)
        self.assertIn("[主库整改](#phase-1)", content)
        self.assertIn("```sql", content)
        self.assertIn(
            "ALTER SESSION SET CONTAINER=CDB$ROOT;\n"
            "ALTER DATABASE ARCHIVELOG;",
            content,
        )

    def test_markdown_projection_uses_safe_fence_length(self) -> None:
        payload = _historical_payload()
        payload["phases"][0]["steps"][0]["commands"][0]["content"] = (
            "echo '```'"
        )

        content = render_implementation_runbook_markdown(payload)

        self.assertIn("````sql\necho '```'\n````", content)

    def test_manual_item_is_not_rendered_as_an_implementation_command(self) -> None:
        payload = _historical_payload()
        payload["phases"][0]["steps"][0]["commands"] = [{
            "command_type": "MANUAL",
            "executor": "MANUAL",
            "title": "确认维护窗口",
            "content": "由业务负责人确认停机窗口。",
        }]

        content = render_implementation_runbook_markdown(payload)

        self.assertIn("### 人工确认项", content)
        self.assertNotIn("### 实施命令", content)
        self.assertNotIn("MANUAL ·", content)
        self.assertNotIn("```text", content)
        self.assertIn("由业务负责人确认停机窗口。", content)
        self.assertTrue(render_implementation_runbook_pdf(payload).startswith(b"%PDF-"))

    def test_command_text_is_not_physically_wrapped(self) -> None:
        command = (
            "ALTER SYSTEM SET log_archive_dest_2='SERVICE=testdb_stby "
            "ASYNC NOAFFIRM VALID_FOR=(ONLINE_LOGFILES,PRIMARY_ROLE) "
            "DB_UNIQUE_NAME=testdb_stby' SCOPE=BOTH SID='*';\n"
            "ALTER SYSTEM SET log_archive_dest_state_2=ENABLE SCOPE=BOTH;"
        )

        self.assertEqual(command, _wrapped_code(command))

    def test_shell_visual_wrap_uses_executable_continuation(self) -> None:
        command = "mkdir -p " + " ".join(
            f"/u02/oradata/TESTDB/PDB{index}" for index in range(1, 8)
        )

        rendered = _formatted_code(command, "SHELL", width=72)

        self.assertIn(" \\\n", rendered)
        self.assertNotIn("\n/u02", rendered)

    def test_shell_visual_wrap_preserves_long_comma_separated_value(self) -> None:
        value = (
            "compatible=26.0.0,processes=1000,"
            "db_unique_name=testdb_dgpdb,"
            "remote_login_passwordfile=EXCLUSIVE"
        )

        rendered = _formatted_code(value, "SHELL", width=48)

        self.assertIn(",\\\n", rendered)
        self.assertEqual(value, rendered.replace("\\\n", ""))

    def test_renderer_accepts_historical_payload_without_v2_validation(self) -> None:
        content = render_implementation_runbook_pdf(_historical_payload())

        self.assertTrue(content.startswith(b"%PDF-"))
        self.assertGreater(len(content), 1000)
        self.assertIn(b"/Outlines", content)

    def test_service_downloads_authorized_active_runbook(self) -> None:
        conversation_id, turn_id = uuid7(), uuid7()
        uow = SimpleNamespace(
            conversations=SimpleNamespace(get_conversation=AsyncMock(
                return_value=SimpleNamespace(created_by="user-1")
            )),
            turns=SimpleNamespace(
                get_turn=AsyncMock(
                    return_value=SimpleNamespace(conversation_id=conversation_id)
                ),
                list_answer_blocks=AsyncMock(return_value=[
                    SimpleNamespace(
                        block_type="IMPLEMENTATION_RUNBOOK",
                        payload_json=_historical_payload(),
                    )
                ]),
            ),
        )
        service = ConversationTurnService(uow_factory=lambda: _context(uow))

        content = asyncio.run(service.get_implementation_runbook_pdf(
            domain_id=1,
            conversation_id=conversation_id,
            turn_id=turn_id,
            actor_id="user-1",
        ))

        self.assertTrue(content.startswith(b"%PDF-"))

        markdown = asyncio.run(service.get_implementation_runbook_markdown(
            domain_id=1,
            conversation_id=conversation_id,
            turn_id=turn_id,
            actor_id="user-1",
        ))
        self.assertTrue(markdown.startswith(b"# "))
        self.assertIn(b"```sql", markdown)

    def test_service_downloads_materialized_script_zip(self) -> None:
        conversation_id, turn_id, run_id = uuid7(), uuid7(), uuid7()
        runbook = compile_implementation_runbook(
            profile=ImplementationProfile.ORACLE_RMAN_BACKUP_BUILD,
            evidence=(),
            context={
                "implementation_parameters": {
                    "INSTANCE_NAME": "TESTDB",
                    "ORACLE_HOME": "/u01/app/oracle/product/26ai/dbhome_1",
                    "BACKUP_DEST": "/backup/testdb",
                }
            },
        )
        generated = generated_artifact_payloads(runbook)
        uow = SimpleNamespace(
            conversations=SimpleNamespace(get_conversation=AsyncMock(
                return_value=SimpleNamespace(created_by="user-1")
            )),
            turns=SimpleNamespace(
                get_turn=AsyncMock(
                    return_value=SimpleNamespace(conversation_id=conversation_id)
                ),
                list_answer_blocks=AsyncMock(return_value=[SimpleNamespace(
                    block_type="IMPLEMENTATION_RUNBOOK",
                    payload_json=runbook.model_dump(mode="json"),
                )]),
                get_run_link=AsyncMock(
                    return_value=SimpleNamespace(ops_run_id=run_id)
                ),
            ),
            runs=SimpleNamespace(list_artifacts=AsyncMock(return_value=[
                SimpleNamespace(
                    schema_version="AIOPS_RUNBOOK_ARTIFACT.v1",
                    payload_json={
                        "artifact_id": item["artifact_id"],
                        "relative_path": item["relative_path"],
                        "media_type": item["media_type"],
                        "content": item["content"],
                    },
                )
                for item in generated
            ])),
        )
        service = ConversationTurnService(uow_factory=lambda: _context(uow))

        content = asyncio.run(service.get_implementation_runbook_zip(
            domain_id=1,
            conversation_id=conversation_id,
            turn_id=turn_id,
            actor_id="user-1",
        ))

        self.assertTrue(content.startswith(b"PK"))
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            self.assertIn("README.md", archive.namelist())
            readme = archive.read("README.md").decode("utf-8")
            self.assertIn("ZIP 可执行包说明", readme)
            self.assertIn("bin/check-last-backup.sh", readme)

    def test_service_rejects_foreign_conversation_and_missing_runbook(self) -> None:
        conversation_id, turn_id = uuid7(), uuid7()
        foreign_uow = SimpleNamespace(
            conversations=SimpleNamespace(get_conversation=AsyncMock(
                return_value=SimpleNamespace(created_by="user-2")
            )),
            turns=SimpleNamespace(
                get_turn=AsyncMock(
                    return_value=SimpleNamespace(conversation_id=conversation_id)
                ),
                list_answer_blocks=AsyncMock(return_value=[]),
            ),
        )
        foreign_service = ConversationTurnService(
            uow_factory=lambda: _context(foreign_uow)
        )
        with self.assertRaises(AIOpsApplicationError):
            asyncio.run(foreign_service.get_implementation_runbook_pdf(
                domain_id=1,
                conversation_id=conversation_id,
                turn_id=turn_id,
                actor_id="user-1",
            ))

        missing_uow = SimpleNamespace(
            conversations=SimpleNamespace(get_conversation=AsyncMock(
                return_value=SimpleNamespace(created_by="user-1")
            )),
            turns=SimpleNamespace(
                get_turn=AsyncMock(
                    return_value=SimpleNamespace(conversation_id=conversation_id)
                ),
                list_answer_blocks=AsyncMock(return_value=[]),
            ),
        )
        missing_service = ConversationTurnService(
            uow_factory=lambda: _context(missing_uow)
        )
        with self.assertRaises(AIOpsApplicationError):
            asyncio.run(missing_service.get_implementation_runbook_pdf(
                domain_id=1,
                conversation_id=conversation_id,
                turn_id=turn_id,
                actor_id="user-1",
            ))


if __name__ == "__main__":
    unittest.main()
