"""用真实仓储验证巡检能力门禁，避免宽松 Mock 掩盖缺失接口。"""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from sqlalchemy.dialects import oracle

from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.inspections.scope import require_inspection_agent_scope
from aiops_agent.entities import AIOpsAgentVersionEntity
from aiops_agent.repositories.agent import AIOpsAgentRepository
from platform_core.identity import uuid7


class InspectionAgentCapabilitiesTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.version_id, self.system_id = uuid7(), uuid7()
        self.session = SimpleNamespace(scalar=AsyncMock())
        self.repository = AIOpsAgentRepository(self.session, self.fail)
        self.uow = SimpleNamespace(agents=self.repository)

    async def require_scope(self):
        await require_inspection_agent_scope(
            uow=self.uow, agent_version_id=self.version_id,
            business_system_id=self.system_id,
        )

    async def test_real_repository_accepts_authorized_active_diagnosis(self):
        self.session.scalar.side_effect = [self.system_id, ["MONITORING_QUERY", "ACTIVE_DIAGNOSIS"]]
        await self.require_scope()
        self.assertEqual(2, self.session.scalar.await_count)
        statement = self.session.scalar.await_args_list[1].args[0]
        self.assertTrue(statement.selected_columns[0].shares_lineage(AIOpsAgentVersionEntity.capabilities_json))
        query = statement.compile(dialect=oracle.dialect())
        self.assertIn(self.version_id, query.params.values())

    async def test_missing_empty_or_other_capabilities_fail_closed(self):
        for capabilities in (None, [], ["MONITORING_QUERY"]):
            with self.subTest(capabilities=capabilities):
                self.session.scalar.side_effect = [self.system_id, capabilities]
                with self.assertRaises(AIOpsApplicationError) as caught:
                    await self.require_scope()
                self.assertEqual("AIOPS_INSPECTION_AGENT_CAPABILITY_REQUIRED", caught.exception.code)
                self.assertEqual(422, caught.exception.status_code)

    async def test_revoked_system_scope_is_rejected_before_capability_read(self):
        self.session.scalar.return_value = None
        with self.assertRaises(AIOpsApplicationError) as caught:
            await self.require_scope()
        self.assertEqual("AIOPS_INSPECTION_SYSTEM_FORBIDDEN", caught.exception.code)
        self.session.scalar.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
