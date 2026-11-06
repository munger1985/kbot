"""重试保留冻结来源；无关新轮次不影响，来源反证与撤权继续阻断。"""

import asyncio
from copy import deepcopy

import pytest

from aiops_agent.application.output_job_sources import OutputJobSourceValidator
from aiops_agent.application.errors import AIOpsApplicationError
from tests.unit.aiops_agent.test_aiops_output_finalization import accepted_before_registry


def test_case_retry_checks_frozen_source_without_model_or_kc_calls():
    async def run():
        f = await accepted_before_registry(retryable=True)
        f.later.turn.status = "RUNNING"
        f.service._documents.stage.reset_mock()
        f.service._structured_model.generate_structured.reset_mock()
        await OutputJobSourceValidator().retry(f.uow, f.job)
        f.service._documents.stage.assert_not_awaited()
        f.service._structured_model.generate_structured.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize("failure", ["hash", "source", "owner", "revoked", "original",
    "invalid_closure_id", "invalid_recovery"])
def test_retry_rejects_changed_envelope_owner_or_actual_source(failure):
    async def run():
        f = await accepted_before_registry(retryable=True)
        f.job.source_json = deepcopy(f.job.input_snapshot_json["source"])
        if failure == "hash":
            f.job.input_hash = "f" * 64
        if failure == "source":
            f.job.source_json["kind"] = "REPORT"
        if failure == "owner":
            f.job.actor_id = "other"
        if failure == "revoked":
            f.uow.agents.version_has_target.return_value = False
        if failure == "original":
            f.artifact.content_hash = "f" * 64
        if failure == "invalid_closure_id":
            f.job.processing_ref_json["closure_artifact_id"] = "not-an-id"
        if failure == "invalid_recovery":
            f.job.processing_ref_json = ["not-a-mapping"]
        with pytest.raises(AIOpsApplicationError):
            await OutputJobSourceValidator().retry(f.uow, f.job)
    asyncio.run(run())
