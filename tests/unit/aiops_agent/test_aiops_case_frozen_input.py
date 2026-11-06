"""受理／执行分离的冻结来源回归，不将 KC 受理视为草稿索引完成。"""

import asyncio
from copy import deepcopy
from datetime import timedelta
import inspect

import pytest

from aiops_agent.application.case_closure import (
    case_from_frozen_input, conversation_case_context, execute_prepared_conversation_case,
    freeze_conversation_case_input, prepare_conversation_case,
)
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.contracts.knowledge import DiagnosisClosureCommand
from aiops_agent.playbooks import canonical_hash
from platform_core.contracts.aiops.output_jobs import FrozenOutputInput
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_conversation_case_closure import preview, refute_first_claim, setup


async def prepare(f, *, included=None):
    candidate = await preview(f, included)
    command = DiagnosisClosureCommand(included_turn_ids=candidate["included_turn_ids"],
        source_fingerprint=candidate["source_fingerprint"], display_name="提交时的案例标题",
        symptoms=("短时业务等待",), root_cause_summary="提交时的人工归纳",
        applicability=("核对现场指标后使用",), contraindications=("不得作为自动执行指令",))
    prepared = await prepare_conversation_case(uow=f.uow, domain_id=f.domain,
        conversation_id=f.conversation_id, actor_id=f.actor, command=command)
    frozen = await freeze_conversation_case_input(prepared=prepared, uow=f.uow, domain_id=f.domain)
    return prepared, frozen


async def execute(f, prepared):
    tenant_id = uuid7() if "tenant_id" in inspect.signature(f.service.close_diagnosis).parameters else None
    return await execute_prepared_conversation_case(f.service, domain_id=f.domain,
        prepared=prepared, context=f.context,
        entity_scope={"tenant_id": tenant_id} if tenant_id else {},
        kc_scope={"tenant_id": tenant_id} if tenant_id else {"domain_id": f.domain},
        model_catalog=f.service._model_catalog if tenant_id else f.service._llm_model_catalog,
        model_scope={"tenant_id": str(tenant_id)} if tenant_id else {},
        technical_name_field="model_name" if tenant_id else "served_model_name")


def test_freezing_is_database_only_and_persisted_input_has_no_runtime_identity():
    async def run():
        f = await setup()
        prepared, frozen = await prepare(f)
        assert frozen.source.source_id == f.conversation_id
        assert frozen.parameters["authority"] == {"origin": "HUMAN", "source_agent_ids": [str(f.run.agent_id)]}
        assert frozen.parameters["case"]["title"] == "提交时的案例标题"
        assert "closure_artifact_id" not in frozen.parameters["case"]
        assert "closed_at" not in frozen.parameters["case"]["closure"]
        assert all(ref.artifact_id in f.state.artifacts for ref in frozen.artifacts)
        assert prepared.closure.source_turn_ids == (f.turn_id, f.later.turn_id)
        f.service._structured_model.generate_structured.assert_not_awaited()
        f.documents.stage.assert_not_awaited()
        f.documents.publish.assert_not_awaited()
        f.service._case_configuration.assert_not_awaited()
        restored = FrozenOutputInput.model_validate(frozen.model_dump(mode="json"))
        assert case_from_frozen_input(restored, closure_artifact_id=prepared.closure_artifact_id) == prepared
    asyncio.run(run())


def test_repeated_acceptance_clock_and_allocated_artifact_ids_do_not_change_business_input_hash():
    async def run():
        f = await setup()
        first, frozen = await prepare(f)
        f.uow.runs.database_now.return_value = first.closure.closed_at + timedelta(seconds=10)
        second, repeated = await prepare(f)
        assert first.closure_artifact_id != second.closure_artifact_id
        assert first.closure.closed_at != second.closure.closed_at
        semantic = lambda value: canonical_hash(value.model_dump(mode="json", exclude={"frozen_at"}))
        assert semantic(frozen) == semantic(repeated)
    asyncio.run(run())


def test_frozen_execution_ignores_unrelated_new_running_turn_but_new_submission_still_rejects():
    async def run():
        f = await setup()
        prepared, frozen = await prepare(f, included=(f.turn_id,))
        f.later.turn.status = "RUNNING"
        f.service._conversation_case_context = None
        restored = case_from_frozen_input(frozen, closure_artifact_id=prepared.closure_artifact_id)
        result = await execute(f, restored)
        assert result["version"]["profile"]["source"]["turn_ids"] == [str(f.turn_id)]
        assert result["version"]["profile"]["root_cause"]["summary"] == "提交时的人工归纳"
        assert result["version"]["status"] == "PROCESSING"
        assert result["asset"]["current_version_id"] is None
        f.documents.stage.assert_awaited_once()
        assert "提交时的案例标题" in f.documents.stage.await_args.kwargs["body"].decode()
        candidate = await conversation_case_context(uow=f.uow, domain_id=f.domain,
            conversation_id=f.conversation_id, actor_id=f.actor, included_turn_ids=(f.turn_id,))
        assert not candidate["eligible"]
        assert any(reason["code"] == "CONVERSATION_ACTIVE" for reason in candidate["reasons"])
    asyncio.run(run())


@pytest.mark.parametrize("change", ["original", "refuted", "permission"])
def test_frozen_inputs_do_not_bypass_real_source_correction_or_revocation(change):
    async def run():
        f = await setup()
        if change == "refuted":
            refute_first_claim(f)
        prepared, frozen = await prepare(f, included=(f.later.turn_id,))
        if change == "original":
            f.later.artifact.payload_json["changed"] = True
        elif change == "permission":
            f.uow.agents.version_has_target.return_value = False
        else:
            # 修改已冻结原件中的反证关系也必须失败，不能用旧指纹绕过。
            f.later.artifact.payload_json["investigation_record"]["claim_revisions"] = []
            f.later.artifact.content_hash = canonical_hash(f.later.artifact.payload_json)
        with pytest.raises(AIOpsApplicationError):
            await execute(f, case_from_frozen_input(frozen, closure_artifact_id=prepared.closure_artifact_id))
        f.service._structured_model.generate_structured.assert_not_awaited()
        f.documents.stage.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize("change", ["fingerprint", "source", "artifact", "request"])
def test_inconsistent_frozen_case_manifest_never_enters_execution(change):
    async def run():
        f = await setup()
        prepared, frozen = await prepare(f)
        data = deepcopy(frozen.model_dump(mode="json"))
        if change == "fingerprint":
            data["source_fingerprint"] = "f" * 64
        elif change == "source":
            data["source"]["source_id"] = str(uuid7())
        elif change == "artifact":
            data["artifacts"][0]["content_hash"] = "f" * 64
        else:
            data["parameters"]["case"]["request_hash"] = "f" * 64
        with pytest.raises(AIOpsApplicationError):
            case_from_frozen_input(FrozenOutputInput.model_validate(data), closure_artifact_id=prepared.closure_artifact_id)
        f.service._structured_model.generate_structured.assert_not_awaited()
    asyncio.run(run())
