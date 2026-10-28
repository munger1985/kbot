"""规划前期限收敛、人工等待和预算守恒的跨仓回归。"""

import asyncio
from datetime import UTC, datetime, timedelta
from functools import wraps
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.dialects import oracle, postgresql

from aiops_agent.application.turn_budget import (
    expire_unstarted_turn, parse_turn_deadline, pause_turn_budget, resume_turn_budget,
)
from aiops_agent.application.turn_queue import TurnQueueService
from aiops_agent.contracts.turn_budget import PausedTurnBudget
from aiops_agent.repositories.turn import TurnRepository


NOW = datetime(2026, 10, 28, 3, tzinfo=UTC)


def run_async(function):
    @wraps(function)
    def execute(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))
    return execute


def test_clarification_preserves_exact_remaining_budget_across_repeated_waits():
    first = pause_turn_budget(NOW + timedelta(seconds=61, microseconds=789123), now=NOW)
    resumed = parse_turn_deadline(resume_turn_budget(first, now=NOW + timedelta(hours=4)))
    assert resumed == NOW + timedelta(hours=4, seconds=61, milliseconds=789)
    second = pause_turn_budget(resumed, now=NOW + timedelta(hours=4, seconds=20))
    assert second["remaining_milliseconds"] == 41789
    again = parse_turn_deadline(resume_turn_budget(second, now=NOW + timedelta(hours=8)))
    assert again == NOW + timedelta(hours=8, milliseconds=41789)


@pytest.mark.parametrize("offset", [-1, 86400, 86401])
def test_wait_ttl_and_clock_reversal_cannot_renew_budget(offset):
    frozen = pause_turn_budget(NOW + timedelta(seconds=60), now=NOW)
    with pytest.raises(ValueError):
        resume_turn_budget(frozen, now=NOW + timedelta(seconds=offset))


def test_exhausted_or_malformed_budget_fails_closed():
    assert pause_turn_budget(None, now=NOW) is None
    with pytest.raises(ValueError):
        parse_turn_deadline("2026-10-28T03:00:00")
    for deadline in [NOW, NOW - timedelta(seconds=1)]:
        with pytest.raises(ValueError):
            resume_turn_budget(pause_turn_budget(deadline, now=NOW), now=NOW)
    for change in [
        {"remaining_milliseconds": True}, {"remaining_milliseconds": -1},
        {"remaining_milliseconds": 86400001}, {"paused_at": NOW.replace(tzinfo=None)},
        {"resume_until": NOW + timedelta(days=2)}, {"extra": 1},
    ]:
        with pytest.raises(ValueError):
            PausedTurnBudget.model_validate({**pause_turn_budget(NOW + timedelta(seconds=60), now=NOW), **change})


def turn(status="ACCEPTED", internal=None):
    return SimpleNamespace(turn_id="t1", status=status, orchestration_internal_json=internal,
        event_cursor=3, error_code=None, completed_at=None, created_at=NOW, domain_id="d1",
        conversation_id="c1")


def uow_for(row, links=()):
    turns = SimpleNamespace(list_run_links=AsyncMock(return_value=list(links)),
        add_event=AsyncMock(), get_turn=AsyncMock(return_value=row))
    return SimpleNamespace(turns=turns)


@pytest.mark.parametrize("status", ["QUEUED", "ACCEPTED", "UNDERSTANDING", "PLANNING"])
@run_async
async def test_expired_unstarted_turn_emits_one_terminal_event(status):
    row = turn(status)
    uow = uow_for(row)
    payload = {"execution_context": {"deadline_at": NOW.isoformat()}}
    assert await expire_unstarted_turn(uow=uow, turn=row, payload=payload, now=NOW)
    assert row.status == "FAILED"
    assert row.error_code == "TURN_DEADLINE_EXCEEDED"
    assert row.event_cursor == 4
    assert row.completed_at == NOW
    event = uow.turns.add_event.call_args.args[0]
    assert event.event_key == "turn.deadline:t1"
    assert not await expire_unstarted_turn(uow=uow, turn=row, payload=payload, now=NOW)
    uow.turns.add_event.assert_awaited_once()


@run_async
async def test_any_run_link_excludes_turn_from_pre_run_reaper():
    row = turn()
    uow = uow_for(row, links=[SimpleNamespace(purpose="RESOURCE_DIAGNOSIS")])
    assert not await expire_unstarted_turn(uow=uow, turn=row,
        payload={"execution_context": {"deadline_at": NOW.isoformat()}}, now=NOW)
    assert row.status == "ACCEPTED"
    uow.turns.add_event.assert_not_awaited()


@run_async
async def test_resumed_deadline_overrides_original_creation_deadline():
    row = turn(internal={"planning_deadline_at": (NOW + timedelta(seconds=30)).isoformat()})
    assert not await expire_unstarted_turn(uow=uow_for(row), turn=row,
        payload={"execution_context": {"deadline_at": (NOW - timedelta(hours=4)).isoformat()}}, now=NOW)


@run_async
async def test_waiting_user_uses_frozen_ttl_not_machine_deadline():
    frozen = pause_turn_budget(NOW + timedelta(seconds=40), now=NOW)
    row = turn("WAITING_USER", {"paused_turn_budget": frozen})
    uow = uow_for(row)
    payload = {"execution_context": {"deadline_at": NOW.isoformat()}}
    assert not await expire_unstarted_turn(uow=uow, turn=row, payload=payload, now=NOW + timedelta(hours=4))
    assert await expire_unstarted_turn(uow=uow, turn=row, payload=payload, now=NOW + timedelta(days=1))


@run_async
async def test_invalid_deadline_is_terminal_but_missing_deadline_is_not_invented():
    row = turn()
    assert not await expire_unstarted_turn(uow=uow_for(row), turn=row, payload={}, now=NOW)
    assert await expire_unstarted_turn(uow=uow_for(row), turn=row,
        payload={"execution_context": {"deadline_at": "bad"}}, now=NOW)
    assert row.error_code == "TURN_BUDGET_INVALID"


@run_async
async def test_repository_filters_any_run_and_uses_portable_bounded_cursor():
    session = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(scalars=lambda: [])))
    repository = TurnRepository(session)
    await repository.list_unstarted_budget_candidates(after=(NOW, "t1"), limit=1000)
    statement = session.execute.call_args.args[0]
    for dialect in [postgresql.dialect(), oracle.dialect()]:
        sql = str(statement.compile(dialect=dialect))
        assert "EXISTS" in sql
        assert "purpose =" not in sql.lower()
        assert "ORDER BY" in sql
    assert statement._limit_clause.value == 32


@run_async
async def test_reaper_rechecks_locked_state_and_releases_each_conversation_lock():
    row = turn()
    rows = [row, SimpleNamespace(**{**vars(row), "turn_id": "t2", "status": "COMPLETED"})]
    uow = uow_for(rows[1])
    uow.turns.list_unstarted_budget_candidates = AsyncMock(side_effect=[rows, []])
    uow.conversations = SimpleNamespace(get_conversation=AsyncMock(return_value=object()))
    uow.outbox = SimpleNamespace(get_by_idempotency=AsyncMock(return_value=SimpleNamespace(
        payload_json={"execution_context": {"deadline_at": NOW.isoformat()}})))
    uow.commit = AsyncMock()

    class Context:
        active = 0
        entered = 0

        async def __aenter__(self):
            assert self.active == 0
            self.active += 1
            self.entered += 1
            return uow

        async def __aexit__(self, *_):
            self.active -= 1

    context = Context()
    queue = TurnQueueService(uow_factory=lambda: context)
    assert not await queue.expire_next()
    assert context.entered == 3
    assert queue._budget_cursor == (NOW, "t2")
    assert not await queue.expire_next()
    assert queue._budget_cursor is None
    uow.commit.assert_not_awaited()
