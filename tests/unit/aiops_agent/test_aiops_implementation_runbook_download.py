"""数据库实施 Runbook PDF 下载边界测试。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.implementation.pdf import (
    _wrapped_code,
    render_implementation_runbook_pdf,
)
from aiops_agent.application.turns import ConversationTurnService
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
    def test_command_text_is_not_physically_wrapped(self) -> None:
        command = (
            "ALTER SYSTEM SET log_archive_dest_2='SERVICE=testdb_stby "
            "ASYNC NOAFFIRM VALID_FOR=(ONLINE_LOGFILES,PRIMARY_ROLE) "
            "DB_UNIQUE_NAME=testdb_stby' SCOPE=BOTH SID='*';"
        )

        self.assertEqual(command, _wrapped_code(command))

    def test_renderer_accepts_historical_payload_without_v2_validation(self) -> None:
        content = render_implementation_runbook_pdf(_historical_payload())

        self.assertTrue(content.startswith(b"%PDF-"))
        self.assertGreater(len(content), 1000)

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
