"""待审批任务不能领取；批准冻结档位，拒绝没有解析快照。"""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from knowledge_algorithms.contracts.errors import AlgorithmError
from knowledge_algorithms.parsing.contracts import LayoutProfile, ParseProfile, ParseStrategy
from knowledge_retrieval_app.repositories.knowledge_ingestion import KnowledgeIngestionRepository


VERSION_ID = UUID("0193e7d9-09b0-7c74-8479-98ef4650b5da")


@pytest.mark.parametrize("approve", [True, False])
def test_unapproved_job_is_not_claimable_and_decision_is_terminal(approve):
    profile = ParseProfile("1", "1", ("zh",), 1024, 8, 8, 1024, 1024, 4, 30, 0,
                           (), 0.1, 0.8, False, LayoutProfile(0.8, 0.05))
    approved = profile.for_approval(ParseStrategy.TEXT) if approve else None
    now = datetime.now(timezone.utc)
    version = SimpleNamespace(document_version_id=VERSION_ID)
    job = SimpleNamespace(document_version_id=VERSION_ID, status="AWAITING_APPROVAL", parse_run_id=VERSION_ID,
                          lease_until=now, lease_owner=None, compute_ref=None)
    run = SimpleNamespace(status="AWAITING_APPROVAL", profile_hash=profile.profile_hash)
    repos = {
        "knowledge_document_version": SimpleNamespace(get=AsyncMock(return_value=version)),
        "knowledge_ingestion_job": SimpleNamespace(get=AsyncMock(return_value=job)),
        "knowledge_parse_run": SimpleNamespace(get=AsyncMock(return_value=run)),
    }
    repository = KnowledgeIngestionRepository(SimpleNamespace(flush=AsyncMock()))
    repository._repos = lambda scope: repos

    async def workflow():
        with pytest.raises(AlgorithmError, match="任务不可重复领取"):
            await repository.claim(None, VERSION_ID, worker_id="worker", now=now, lease_seconds=20)
        await repository.decide_approval(None, VERSION_ID, document_version_id=VERSION_ID, profile=approved, now=now)
        assert run.status == job.status == ("RECEIVED" if approve else "REJECTED")
        if approve:
            assert run.profile_hash == approved.profile_hash
            assert run.profile_json["strategy"] == "TEXT"
            assert run.model_snapshot_json == {}
        else:
            assert not hasattr(run, "profile_json")
            with pytest.raises(AlgorithmError, match="任务不可重复领取"):
                await repository.claim(None, VERSION_ID, worker_id="worker", now=now, lease_seconds=20)
        with pytest.raises(AlgorithmError, match="当前任务已不处于待审批状态"):
            await repository.decide_approval(None, VERSION_ID, document_version_id=VERSION_ID, profile=approved, now=now)

    asyncio.run(workflow())
