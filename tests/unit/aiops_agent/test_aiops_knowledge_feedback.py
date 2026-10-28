"""复用反馈关联精确版本，人工意见不得提升知识验证状态。"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from aiops_agent.application.operations_knowledge import OperationsKnowledgeService
from aiops_agent.application.errors import AIOpsApplicationError
from platform_core.contracts.aiops.knowledge import KnowledgeReuseFeedback
from platform_core.identity import uuid7


@pytest.mark.parametrize("effect", ["IMPROVED", "UNCHANGED", "DEGRADED", "NOT_ATTEMPTED"])
def test_feedback_is_idempotent_redacted_user_report_without_status_change(effect):
    async def run():
        actor, asset_id, version_id, tenant, domain = (uuid7() for _ in range(5))
        asset = SimpleNamespace(asset_id=asset_id, tenant_id=tenant, domain_id=domain, asset_kind="DIAGNOSIS_CASE",
            status="PUBLISHED", current_version_id=version_id)
        version = SimpleNamespace(asset_id=asset_id, asset_version_id=version_id, status="PUBLISHED", profile_json={"case_kind": "DIAGNOSTIC_REFERENCE"})
        repo = SimpleNamespace(get_asset=AsyncMock(return_value=asset), get_version=AsyncMock(return_value=version),
            get_review=AsyncMock(return_value=None), add_review=AsyncMock())
        uow = SimpleNamespace(operations_knowledge=repo, commit=AsyncMock())
        class Uow:
            async def __aenter__(self): return uow
            async def __aexit__(self, *args): pass
        service = OperationsKnowledgeService.__new__(OperationsKnowledgeService)
        service._uow_factory, service._actor = Uow, lambda _: actor
        command = KnowledgeReuseFeedback(feedback_id=uuid7(), asset_version_id=version_id,
            applicability="APPLICABLE", action_effect=effect, notes="10.0.0.88 有改善 token=secret")
        receipt = await service.record_reuse_feedback(domain_id=domain, asset_id=asset_id, command=command, context=object())
        assert receipt["trust_level"] == "USER_PROVIDED" and not receipt["replayed"]
        review = repo.add_review.await_args.args[0]
        assert review.before_status == review.after_status == "PUBLISHED"
        assert "USER_REPORTED" in review.comment_text and "10.0.0.88" not in review.comment_text and "secret" not in review.comment_text.replace("<secret_1>", "")
        assert asset.current_version_id == version_id and version.profile_json["case_kind"] == "DIAGNOSTIC_REFERENCE"
        repo.get_review.return_value = review
        receipt = await service.record_reuse_feedback(domain_id=domain, asset_id=asset_id, command=command, context=object())
        assert receipt["replayed"]
        repo.add_review.assert_awaited_once()
        with pytest.raises(AIOpsApplicationError):
            await service.record_reuse_feedback(domain_id=domain, asset_id=asset_id,
                command=command.model_copy(update={"notes": "不同内容"}), context=object())
        assert uow.commit.await_count == 1
    asyncio.run(run())


def test_feedback_rejects_blank_notes_and_arbitrary_verification_flags():
    payload = {"feedback_id": uuid7(), "asset_version_id": uuid7(), "applicability": "UNCERTAIN", "notes": " "}
    with pytest.raises(ValidationError):
        KnowledgeReuseFeedback.model_validate(payload)
    with pytest.raises(ValidationError):
        KnowledgeReuseFeedback.model_validate({**payload, "notes": "说明", "verified": True})
