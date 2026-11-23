"""Oracle 不依赖业务唯一约束，运行表锁内完成认领与去重。"""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from aiops_agent.entities.data_protection import ProtectionRunEntity
from aiops_agent.repositories.data_protection import ProtectionPlanRepository


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reserve", "occupied", "inserts"),
    [(True, True, False), (True, False, True), (False, True, False), (False, False, True)],
)
async def test_trigger_and_destination_are_checked_under_oracle_transaction_lock(reserve, occupied, inserts):
    row = ProtectionRunEntity(
        resource_id=uuid4(), operation="DRILL", scheduled_at=datetime.now(UTC),
        active_key="shared-destination" if reserve else None,
    )
    previous = object() if occupied else None
    session = SimpleNamespace(
        execute=AsyncMock(side_effect=[None, SimpleNamespace(scalar_one_or_none=lambda: previous)]),
        add=Mock(), flush=AsyncMock(),
    )
    repo = ProtectionPlanRepository(session)

    if reserve:
        assert await repo.reserve(row) is inserts
    else:
        assert await repo.add_run(row) is (row if inserts else previous)

    lock, lookup = (call.args[0] for call in session.execute.await_args_list)
    assert "SHARE ROW EXCLUSIVE MODE WAIT 5" in str(lock)
    condition = str(lookup).split("WHERE", 1)[1].lower()
    assert "resource_id" in condition and "operation" in condition and "scheduled_at" in condition
    assert ("active_key" in condition) is reserve
    assert session.add.call_count == session.flush.await_count == int(inserts)
