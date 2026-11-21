"""已持久化的解析失败或取消不得被流水线包装为合同错误。"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from knowledge_retrieval_app.application.knowledge_pipeline import KnowledgePipelineWorker


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["FAILED", "CANCELLED"])
@pytest.mark.parametrize("already_terminal", [False, True])
async def test_parse_only_preserves_terminal_status_without_indexing(status, already_terminal):
    now = datetime.now(timezone.utc)
    active = SimpleNamespace(status="PARSING", lease_owner="worker",
                             lease_until=now + timedelta(seconds=360),
                             profile=SimpleNamespace(timeout_seconds=300))
    terminal = SimpleNamespace(status=status)
    ingestion = SimpleNamespace(
        get_job=AsyncMock(side_effect=[terminal] if already_terminal else [active, terminal]),
        poll=AsyncMock(),
        clock=lambda: now,
    )
    projection = Mock()
    worker = KnowledgePipelineWorker(ingestion=ingestion, projection=projection, builder_factory=None,
                                     profile=None, allowed_policy=None, policy_revision=None, worker_id="worker")

    assert await worker.process(object(), object(), parse_only=True) == {"status": status}
    assert ingestion.poll.await_count == (0 if already_terminal else 1)
    projection.allowed_document.assert_not_called()
