"""Agent Runtime 跨数据库时间映射测试。"""

from datetime import datetime, timezone
import unittest

from agent_runtime.entities.agent_definition import AgentDefinitionEntity
from agent_runtime.entities.conversation import AgentConversationEntity
from agent_runtime.entities.runtime import (
    AgentDelegationEntity,
    AgentRunEntity,
    AgentTaskEntity,
)
from platform_core.persistence.orm import UniversalTimestamp


class AgentRuntimeTimestampMappingTest(unittest.TestCase):
    def test_runtime_lease_and_deadline_use_universal_timestamp(self) -> None:
        columns = (
            AgentRunEntity.__table__.c.deadline_at,
            AgentTaskEntity.__table__.c.lease_until,
            AgentTaskEntity.__table__.c.next_retry_at,
            AgentDelegationEntity.__table__.c.lease_until,
            AgentDefinitionEntity.__table__.c.created_at,
            AgentConversationEntity.__table__.c.created_at,
        )

        for column in columns:
            with self.subTest(column=column.name):
                self.assertIsInstance(column.type, UniversalTimestamp)

    def test_oracle_naive_result_is_restored_as_utc(self) -> None:
        timestamp_type = AgentTaskEntity.__table__.c.lease_until.type
        restored = timestamp_type.process_result_value(
            datetime(2026, 7, 27, 9, 53, 35),
            dialect=type("OracleDialect", (), {"name": "oracle"})(),
        )

        self.assertEqual(timezone.utc, restored.tzinfo)
        self.assertEqual(
            datetime(2026, 7, 27, 9, 53, 35, tzinfo=timezone.utc),
            restored,
        )


if __name__ == "__main__":
    unittest.main()
