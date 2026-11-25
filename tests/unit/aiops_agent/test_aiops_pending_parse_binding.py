"""待领取的运维资料解析可换绑，已领取的解析保留原代次。"""

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select

from aiops_agent.entities.knowledge_ingestion_job import KnowledgeIngestionJobEntity
from aiops_agent.repositories.knowledge_documents import OperationsDocumentRepository
from knowledge_algorithms.contracts.wire import encode
from knowledge_algorithms.parsing.contracts import LayoutProfile, ParseModelBinding, ParseProfile, ParseStrategy
from platform_core.identity import uuid7


@pytest.mark.parametrize("status", ["RECEIVED", "PARSING"])
def test_refresh_does_not_replace_a_claimed_parse_attempt(status):
    profile = ParseProfile("1", "1", ("zh",), 1024, 8, 8, 1024, 1024, 4, 30, 0,
                           (), 0.1, 0.8, False, LayoutProfile(0.8, 0.05))
    ocr = ParseModelBinding(str(uuid7()), "2", "ocr:v2")
    vlm = ParseModelBinding(str(uuid7()), "3", "vlm:v3")
    new = profile.for_approval(ParseStrategy.VISUAL, ocr=ocr, vlm=vlm)
    old = profile.for_approval(ParseStrategy.VISUAL, ocr=ParseModelBinding(ocr.model_id, "1", "ocr:v1"), vlm=vlm)
    version_id, old_run_id, new_run_id = uuid7(), uuid7(), uuid7()
    version = SimpleNamespace(status=status, source_sha256="a" * 64, active_generation_id=None)
    job = SimpleNamespace(status=status, parse_run_id=old_run_id, attempt=1, fence=4)
    old_run = SimpleNamespace(status=status, profile_hash=old.profile_hash, profile_json=encode(old))
    runs = SimpleNamespace(get=AsyncMock(return_value=old_run), add=AsyncMock(return_value=SimpleNamespace(
        parse_run_id=new_run_id)))
    session = SimpleNamespace(scalars=AsyncMock(return_value=SimpleNamespace(all=lambda: [job])))
    repo = OperationsDocumentRepository(session)
    repo._repos = lambda _: {
        "knowledge_document_version": SimpleNamespace(get=AsyncMock(return_value=version)),
        "knowledge_ingestion_job": SimpleNamespace(entity=KnowledgeIngestionJobEntity,
                                                  statement=lambda: select(KnowledgeIngestionJobEntity)),
        "knowledge_parse_run": runs,
    }

    refreshed = asyncio.run(repo.retry_document(None, version_id, now=datetime.now(UTC), pending_profile=new))

    if status == "PARSING":
        assert not refreshed
        assert (job.parse_run_id, job.attempt, job.fence, old_run.status) == (old_run_id, 1, 4, "PARSING")
        runs.get.assert_not_awaited()
        runs.add.assert_not_awaited()
    else:
        assert refreshed
        assert (job.parse_run_id, job.attempt, job.fence, old_run.status) == (new_run_id, 2, 5, "SUPERSEDED")
        assert old_run.profile_json == encode(old)
        new_run = runs.add.await_args.args[0]
        assert new_run["source_sha256"] == version.source_sha256
        assert new_run["profile_hash"] == new.profile_hash
        assert new_run["profile_json"] == encode(new)
        assert new_run["model_snapshot_json"] == {"ocr": encode(ocr), "vlm": encode(vlm)}
