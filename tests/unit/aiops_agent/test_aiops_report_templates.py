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
