"""案例后台复核必须轮转、有界，并在同一事务暂停检索与留痕。"""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.application.operations_knowledge import OperationsKnowledgeService
from aiops_agent.repositories.knowledge import OperationsKnowledgeRepository
from platform_core.identity import uuid7


@pytest.mark.parametrize("valid", [True, False])
def test_background_source_validation_withdraws_only_invalid_case(monkeypatch, valid):
    version_id, asset_id, actor, domain, tenant, run_id = (uuid7() for _ in range(6))
    version = SimpleNamespace(asset_version_id=version_id, asset_id=asset_id,
        status="PUBLISHED", profile_json={})
    asset = SimpleNamespace(asset_id=asset_id, current_version_id=version_id, tenant_id=tenant,
        domain_id=domain, created_by=actor, status="PUBLISHED")
    index = SimpleNamespace(asset_version_id=version_id, last_checked_at=None)
    repo = SimpleNamespace(
        next_case_index_for_validation=AsyncMock(return_value=(index, domain)),
        get_version=AsyncMock(return_value=version), get_asset=AsyncMock(return_value=asset),
        add_review=AsyncMock(), get_version_relations=AsyncMock(return_value={
            "sources": [SimpleNamespace(source_run_id=run_id)]}),
    )
    uow = SimpleNamespace(operations_knowledge=repo, commit=AsyncMock(),
        runs=SimpleNamespace(get_run_scoped=AsyncMock(return_value=SimpleNamespace(ops_run_id=run_id))),
        platform_notifications=SimpleNamespace(emit_case_review_required=AsyncMock()))
    class Uow:
        async def __aenter__(self): return uow
        async def __aexit__(self, *args): pass
    service = OperationsKnowledgeService.__new__(OperationsKnowledgeService)
    service._uow_factory = Uow
    monkeypatch.setattr("aiops_agent.application.operations_knowledge.validate_case_sources", AsyncMock(return_value=valid))
    monkeypatch.setattr("aiops_agent.application.operations_knowledge.validate_case_effect", AsyncMock(return_value=True))
    assert asyncio.run(service.revalidate_next_case())
    assert index.last_checked_at is not None
    if valid:
        assert version.status == asset.status == "PUBLISHED"
        repo.add_review.assert_not_awaited()
        uow.platform_notifications.emit_case_review_required.assert_not_awaited()
    else:
        assert version.status == asset.status == "REVIEW_REQUIRED"
        assert asset.current_version_id is None
        review = repo.add_review.await_args.args[0]
        assert review.decision == "SOURCE_INVALIDATED"
        assert review.before_status == "PUBLISHED"
        uow.platform_notifications.emit_case_review_required.assert_awaited_once()
    uow.commit.assert_awaited_once()


def test_case_validation_query_locates_first_then_locks_without_oracle_pagination():
    from sqlalchemy.dialects import oracle, postgresql
    selected_id, domain = uuid7(), uuid7()
    index = SimpleNamespace(index_ref_id=selected_id)
    session = SimpleNamespace(execute=AsyncMock(side_effect=[
        SimpleNamespace(first=lambda: (selected_id, domain)),
        SimpleNamespace(scalar_one_or_none=lambda: index),
    ]))
    repo = OperationsKnowledgeRepository(session)
    assert asyncio.run(repo.next_case_index_for_validation(before=datetime.now(timezone.utc))) == (index, domain)
    statements = [call.args[0] for call in session.execute.call_args_list]
    assert statements[0]._limit_clause is not None
    assert statements[0]._for_update_arg is None
    assert statements[1]._limit_clause is None
    assert statements[1]._for_update_arg.skip_locked is True
    for dialect in (oracle.dialect(), postgresql.dialect()):
        assert str(statements[0].compile(dialect=dialect))
        assert "FOR UPDATE" in str(statements[1].compile(dialect=dialect))
