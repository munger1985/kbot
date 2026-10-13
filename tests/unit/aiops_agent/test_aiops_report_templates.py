"""AIOps 报告模板版本更新测试。"""

from __future__ import annotations

import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

from aiops_agent.application.report_templates import (
    InspectionTemplateService,
    SessionReportTemplateService,
)
from aiops_agent.application.errors import AIOpsApplicationError
from platform_core.identity import uuid7


class _UnitOfWorkContext:
    def __init__(self, uow) -> None:
        self._uow = uow

    async def __aenter__(self):
        return self._uow

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback


class _PostCommitGuardedRow:
    """模拟提交后 ORM 服务端更新字段过期的实体。"""

    def __init__(self, *, updated_at: datetime, **attributes) -> None:
        for name, value in attributes.items():
            setattr(self, name, value)
        self._updated_at = updated_at
        self._updated_at_assigned = False
        self._committed = False

    @property
    def updated_at(self) -> datetime:
        if self._committed and not self._updated_at_assigned:
            raise RuntimeError("提交后禁止隐式加载 updated_at")
        return self._updated_at

    @updated_at.setter
    def updated_at(self, value: datetime) -> None:
        self._updated_at = value
        self._updated_at_assigned = True

    def mark_committed(self) -> None:
        self._committed = True


class ReportTemplateVersionUpdateTest(unittest.IsolatedAsyncioTestCase):
    async def test_resolve_freezes_requested_version_owned_by_template(self) -> None:
        template_id = uuid7()
        current_version_id = uuid7()
        requested_version_id = uuid7()
        definition, _ = SessionReportTemplateService._normalize_definition(
            display_name="会话复盘",
            definition={"sections": [{"kind": "FINDINGS"}]},
        )
        row = SimpleNamespace(
            template_id=template_id,
            status="ACTIVE",
            display_name="会话复盘",
            current_version_id=current_version_id,
        )
        version = SimpleNamespace(
            template_version_id=requested_version_id,
            domain_id=7,
            template_id=template_id,
            version_no=2,
            definition_json=definition,
        )
        inspections = SimpleNamespace(
            get_session_report_template=AsyncMock(return_value=row),
            get_session_report_template_version=AsyncMock(return_value=version),
        )
        service = SessionReportTemplateService(
            uow_factory=lambda: _UnitOfWorkContext(
                SimpleNamespace(inspections=inspections)
            )
        )

        result = await service.resolve(
            domain_id=7,
            template_ref=f"domain:{template_id}",
            template_version_id=requested_version_id,
        )

        self.assertEqual("2", result.version)
        inspections.get_session_report_template_version.assert_awaited_once_with(
            template_version_id=requested_version_id
        )

        version.template_id = uuid7()
        with self.assertRaises(AIOpsApplicationError) as caught:
            await service.resolve(
                domain_id=7,
                template_ref=f"domain:{template_id}",
                template_version_id=requested_version_id,
            )
        self.assertEqual("OPS_STATE_CONFLICT", caught.exception.code)

    async def test_set_default_locks_domain_before_switching_template(self) -> None:
        calls = []
        template_id = uuid7()
        row = SimpleNamespace(
            template_id=template_id,
            domain_id=7,
            display_name="会话复盘",
            status="ACTIVE",
            is_default=0,
            current_version_id=uuid7(),
            row_version=1,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            updated_at=datetime(2026, 1, 1, tzinfo=UTC),
            updated_by="actor:old",
        )

        async def lock_domain(*, domain_id):
            calls.append(("lock", domain_id))
            return True

        async def clear_default(*, domain_id, keep_template_id=None):
            calls.append(("clear", domain_id))
            self.assertEqual(template_id, keep_template_id)

        inspections = SimpleNamespace(
            lock_domain=AsyncMock(side_effect=lock_domain),
            clear_default_session_report_template=AsyncMock(
                side_effect=clear_default
            ),
            get_session_report_template=AsyncMock(return_value=row),
        )
        uow = SimpleNamespace(inspections=inspections, commit=AsyncMock())
        service = SessionReportTemplateService(
            uow_factory=lambda: _UnitOfWorkContext(uow)
        )

        result = await service.set_default(
            domain_id=7,
            actor_id="actor:new",
            template_ref=f"domain:{template_id}",
        )

        self.assertEqual([("lock", 7), ("clear", 7)], calls)
        self.assertTrue(result["is_default"])
        uow.commit.assert_awaited_once()

    async def test_definition_requires_protected_sections_at_boundaries(self) -> None:
        with self.assertRaises(AIOpsApplicationError):
            SessionReportTemplateService._normalize_definition(
                display_name="无效模板",
                definition={
                    "sections": [
                        {"kind": "EVIDENCE_BOUNDARY"},
                        {"kind": "FINDINGS"},
                    ]
                },
            )

    async def test_session_template_response_does_not_load_expired_timestamp(
        self,
    ) -> None:
        old_updated_at = datetime(2026, 1, 1, tzinfo=UTC)
        template_id = uuid7()
        row = _PostCommitGuardedRow(
            template_id=template_id,
            domain_id=7,
            display_name="会话复盘",
            status="ACTIVE",
            current_version_id=uuid7(),
            row_version=1,
            updated_by="actor:old",
            updated_at=old_updated_at,
        )

        async def commit() -> None:
            row.row_version = 2
            row.mark_committed()

        inspections = SimpleNamespace(
            get_session_report_template=AsyncMock(return_value=row),
            next_session_report_template_version=AsyncMock(return_value=2),
            add_session_report_template_version=AsyncMock(),
        )
        uow = SimpleNamespace(
            inspections=inspections,
            commit=AsyncMock(side_effect=commit),
        )
        service = SessionReportTemplateService(
            uow_factory=lambda: _UnitOfWorkContext(uow)
        )

        result = await service.create_version(
            domain_id=7,
            actor_id="actor:new",
            template_id=template_id,
            expected_row_version=1,
            definition={"sections": [{"kind": "FINDINGS"}]},
        )

        self.assertEqual(2, result["row_version"])
        self.assertEqual(row.updated_at.isoformat(), result["updated_at"])
        self.assertGreater(row.updated_at, old_updated_at)
        uow.commit.assert_awaited_once()

    async def test_inspection_template_response_does_not_load_expired_timestamp(
        self,
    ) -> None:
        old_updated_at = datetime(2026, 1, 1, tzinfo=UTC)
        template_id = uuid7()
        row = _PostCommitGuardedRow(
            inspection_template_id=template_id,
            display_name="每日巡检",
            status="ACTIVE",
            current_version_id=uuid7(),
            row_version=1,
            updated_by="actor:old",
            updated_at=old_updated_at,
        )

        async def commit() -> None:
            row.row_version = 2
            row.mark_committed()

        inspections = SimpleNamespace(
            get_inspection_template=AsyncMock(return_value=row),
            next_inspection_template_version=AsyncMock(return_value=2),
            add_inspection_template_version=AsyncMock(),
        )
        uow = SimpleNamespace(
            inspections=inspections,
            commit=AsyncMock(side_effect=commit),
        )
        service = InspectionTemplateService(
            uow_factory=lambda: _UnitOfWorkContext(uow)
        )

        result = await service.create_version(
            domain_id=7,
            actor_id="actor:new",
            template_id=template_id,
            expected_row_version=1,
            selected_check_ids=("oracle.backup.rman_volume",),
        )

        self.assertEqual(2, result["row_version"])
        self.assertEqual(row.updated_at.isoformat(), result["updated_at"])
        self.assertGreater(row.updated_at, old_updated_at)
        uow.commit.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
