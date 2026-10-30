"""会话收口的逐轮选择与逐产物血缘，不从人工摘要反推受控诊断或业务效果。"""

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from types import SimpleNamespace
from uuid import UUID

from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.diagnosis.investigation_record import investigation_report_source
from aiops_agent.contracts.investigation import InvestigationRecord
from aiops_agent.contracts.knowledge import DiagnosisClosure, DiagnosisClosureSource
from aiops_agent.playbooks import canonical_hash


@dataclass(frozen=True)
class ClosureArtifactSource:
    source_run_id: UUID
    source_artifact_id: UUID
    content_hash: str
    source_report_id: UUID | None = None


def rejected_closure():
    return AIOpsApplicationError(code="AIOPS_DIAGNOSIS_CLOSURE_REJECTED",
        message="收口来源、有效判断或当前授权已变化，请刷新并重新核验", status_code=422)


def closure_evidence_role(schema):
    if schema in {"EXECUTION_RESULT.v1", "USER_PROVIDED_ACTION_RESULT.v1"}:
        return "ACTION"
    if schema in {"ACTION_VERIFICATION.v1", "COMPARISON_RESULT.v1", "GROUNDING_VERIFICATION.v1"}:
        return "VERIFICATION"
    return "DIAGNOSIS"


def closure_reference(*, turn, run, artifact, roles, claim_ids=()):
    return DiagnosisClosureSource(turn_id=turn.turn_id, run_id=run.ops_run_id,
        artifact_id=artifact.artifact_id, schema_version=artifact.schema_version,
        content_hash=artifact.content_hash, security_level=int(artifact.security_level),
        observed_at=artifact.created_at,
        roles=tuple(roles), claim_ids=tuple(sorted(claim_ids)))


def closure_claim_refs(artifact, claim_ids):
    source = investigation_report_source(InvestigationRecord.model_validate(artifact.payload_json["investigation_record"]))
    return [{key: fact[key] for key in ("source_run_id", "record_id", "record_revision", "claim_id", "evidence_refs")}
        for fact in source["facts"] if fact["claim_id"] in claim_ids and fact.get("level") in {"OBSERVED", "SUPPORTED"}]


async def validate_closure_sources(*, uow, domain_id, run, artifact, agent_id=None, plan_capture_reader=None):
    """原件、轮次归属、当前权限、有效 Claim 和全部动作/窗口引用均须重验。"""
    from aiops_agent.application.case_source_validation import validate_case_source_rows
    try:
        closure = DiagnosisClosure.model_validate(artifact.payload_json)
        target = await uow.targets.get_scoped(target_id=closure.target_id, domain_id=domain_id)
        if (target is None or target.status != "ENABLED" or target.row_version != closure.target_version
            or str(closure.target_id) != str(run.target_id) or canonical_hash(artifact.payload_json) != artifact.content_hash):
            return False
        conversation = await uow.conversations.get_conversation(domain_id=domain_id, conversation_id=closure.conversation_id)
        if conversation is None or str(conversation.created_by) != closure.closed_by:
            return False
        turns = await uow.turns.list_all_turns(conversation_id=closure.conversation_id, limit=201)
        if len(turns) > 200 or any(turn.status not in {"COMPLETED", "PARTIAL", "FAILED", "CANCELLED"} for turn in turns):
            return False
        roots, claim_refs, rows, selection = [], [], [], {}
        role_ids = {role: set() for role in ("INVESTIGATION", "ACTION", "BEFORE", "AFTER")}
        seen = set()
        for ref in closure.source_refs:
            if ref.artifact_id in seen or len(set(ref.roles)) != len(ref.roles):
                return False
            seen.add(ref.artifact_id)
            original = await uow.runs.get_artifact(artifact_id=ref.artifact_id)
            owner = await uow.runs.get_run_scoped(ops_run_id=ref.run_id, domain_id=domain_id)
            turn = await uow.turns.get_turn(domain_id=domain_id, turn_id=ref.turn_id)
            links = await uow.turns.list_run_links(turn_id=ref.turn_id)
            if (original is None or owner is None or turn is None
                or str(turn.conversation_id) != str(closure.conversation_id)
                or str(owner.target_id) != str(closure.target_id)
                or not any(str(link.ops_run_id) == str(ref.run_id) for link in links)
                or str(original.ops_run_id) != str(ref.run_id) or original.schema_version != ref.schema_version
                or original.schema_version == "DIAGNOSIS_CLOSURE.v1"
                or original.content_hash != ref.content_hash or canonical_hash(original.payload_json) != ref.content_hash
                or original.security_level != ref.security_level or original.created_at != ref.observed_at):
                return False
            for role in ref.roles:
                role_ids[role].add(str(ref.artifact_id))
            if "INVESTIGATION" in ref.roles:
                if (original.schema_version != "AIOPS_TURN_RESULT.v1" or not ref.claim_ids
                    or len(set(ref.claim_ids)) != len(ref.claim_ids)
                    or owner.status not in {"COMPLETED", "PARTIAL"} or turn.status not in {"COMPLETED", "PARTIAL"}
                    or owner.final_artifact_id != ref.artifact_id
                    or not any(link.purpose == "PRIMARY" and str(link.ops_run_id) == str(ref.run_id) for link in links)):
                    return False
                roots.append((turn.turn_no, str(ref.turn_id)))
                selection[str(ref.artifact_id)] = set(ref.claim_ids)
                projected = closure_claim_refs(original, ref.claim_ids)
                if {item["claim_id"] for item in projected} != set(ref.claim_ids):
                    return False
                claim_refs.extend(projected)
            elif ref.claim_ids:
                return False
            if "ACTION" in ref.roles and not await closure_action_is_valid(uow=uow, run=owner, artifact=original):
                return False
            rows.append(ClosureArtifactSource(ref.run_id, ref.artifact_id, ref.content_hash))
        ordered_turns = [identity for _, identity in sorted(roots)]
        profile = closure.profile
        if (not ordered_turns or len(set(ordered_turns)) != len(ordered_turns)
            or ordered_turns != [str(value) for value in closure.source_turn_ids]
            or any(str(ref.turn_id) not in ordered_turns for ref in closure.source_refs)
            or role_ids["BEFORE"] & role_ids["AFTER"]
            or profile.source.get("claim_refs") != claim_refs
            or profile.source.get("turn_ids") != ordered_turns
            or set(profile.source.get("run_ids") or ()) != {str(ref.run_id) for ref in closure.source_refs}
            or profile.source.get("conversation_id") != str(closure.conversation_id)
            or profile.source.get("source_fingerprint") != closure.source_fingerprint
            or set(profile.root_cause.get("evidence_refs") or ()) != role_ids["INVESTIGATION"]
            or set(profile.verification.get("before_refs") or ()) != role_ids["BEFORE"]
            or set(profile.verification.get("after_refs") or ()) != role_ids["AFTER"]
            or profile.verification.get("method") != "USER_REPORTED"
            or profile.case_kind not in {"DIAGNOSTIC_REFERENCE", "NEGATIVE_EXPERIENCE"}
            or artifact.security_level < max(ref.security_level for ref in closure.source_refs)
            or str(artifact.ops_run_id) not in {str(ref.run_id) for ref in closure.source_refs if "INVESTIGATION" in ref.roles}
            or any(item.get("source_kind") != "USER_PROVIDED"
                or set(item.get("evidence_refs") or ()) != role_ids["ACTION"] for item in profile.actions)):
            return False
        return await validate_case_source_rows(uow=uow, domain_id=domain_id, sources=rows,
            agent_id=agent_id, plan_capture_reader=plan_capture_reader, claim_selection=selection)
    except (KeyError, TypeError, ValueError):
        return False


async def closure_action_is_valid(*, uow, run, artifact):
    """核对执行登记；人工执行记录仍明确是人工来源，不能凭 schema 冒充执行回执。"""
    if artifact.schema_version == "EXECUTION_RESULT.v1":
        from aiops_agent.contracts.change.models import ExecutionResultArtifact
        receipt = ExecutionResultArtifact.model_validate(artifact.payload_json)
        execution = await uow.changes.get_execution(execution_id=receipt.execution_id)
        return (artifact.trust_level == "SOURCE_VERIFIED" and execution is not None
            and str(execution.result_artifact_id) == str(artifact.artifact_id)
            and str(execution.ops_run_id) == str(run.ops_run_id)
            and str(execution.ops_task_id) == str(artifact.ops_task_id)
            and str(execution.target_id) == str(run.target_id)
            and str(execution.proposal_id) == str(receipt.proposal_id)
            and str(execution.executor_request_id) == str(receipt.executor_request_id)
            and execution.executor_instance_id == receipt.executor_instance_id
            and execution.status_version == receipt.status_version
            and execution.result_hash == receipt.result_hash
            and execution.grant_jti_hash == receipt.grant_jti_hash
            and execution.status == receipt.status
            and execution.proposal_hash == receipt.proposal_hash and execution.command_hash == receipt.command_hash)
    if artifact.schema_version == "USER_PROVIDED_ACTION_RESULT.v1":
        from aiops_agent.contracts.change.models import AdvisoryActionResult
        reported = AdvisoryActionResult.model_validate(artifact.payload_json)
        return (artifact.trust_level == "USER_PROVIDED" and reported.status in {"EXECUTED", "FAILED"}
            and str(reported.submitted_by) == str(run.actor_id))
    return False


async def conversation_case_context(*, uow, domain_id, conversation_id, actor_id,
    included_turn_ids=None, plan_capture_reader=None):
    """选择同一 Target 的有效判断，保留其他轮次不可选原因，不取历史最高等级。"""
    from aiops_agent.application.case_source_validation import current_case_claim_ids, validate_case_source_rows
    conversation = await uow.conversations.get_conversation(conversation_id=conversation_id, domain_id=domain_id)
    if conversation is None or str(conversation.created_by) != str(actor_id):
        raise AIOpsApplicationError(code="AIOPS_RESOURCE_NOT_FOUND", message="会话不存在或无权访问", status_code=404)
    turns = sorted(await uow.turns.list_all_turns(conversation_id=conversation_id, limit=201), key=lambda item: item.turn_no)
    reasons = []
    if len(turns) > 200:
        reasons.append({"code": "CLOSURE_TURN_LIMIT", "message": "会话超过 200 轮，请将同一问题整理为独立调查后收口"})
    if any(turn.status not in {"COMPLETED", "PARTIAL", "FAILED", "CANCELLED"} for turn in turns):
        reasons.append({"code": "CONVERSATION_ACTIVE", "message": "会话仍有进行中的诊断，不能收口"})
    available, options = [], []
    for turn in turns[:200]:
        option = {"turn_id": str(turn.turn_id), "turn_no": int(turn.turn_no), "eligible": False,
            "reason": "本轮未形成可追溯的有效判断", "target_id": None, "summary": "", "selected": False}
        options.append(option)
        link = await uow.turns.get_run_link(turn_id=turn.turn_id, purpose="PRIMARY")
        owner = await uow.runs.get_run_scoped(ops_run_id=link.ops_run_id, domain_id=domain_id) if link else None
        if owner is None or owner.status not in {"COMPLETED", "PARTIAL"} or owner.final_artifact_id is None:
            continue
        option["target_id"] = str(owner.target_id)
        artifact = await uow.runs.get_artifact(artifact_id=owner.final_artifact_id)
        if artifact is None or artifact.schema_version != "AIOPS_TURN_RESULT.v1":
            continue
        try:
            record = InvestigationRecord.model_validate(artifact.payload_json.get("investigation_record"))
            current = await current_case_claim_ids(uow=uow, domain_id=domain_id, run=owner, artifact=artifact, record=record)
            source = investigation_report_source(record)
            claims = {fact["claim_id"] for fact in source["facts"] if fact.get("level") in {"OBSERVED", "SUPPORTED"}}
            claims &= current or set()
            if not claims or not await validate_case_source_rows(uow=uow, domain_id=domain_id,
                sources=[ClosureArtifactSource(owner.ops_run_id, artifact.artifact_id, artifact.content_hash)],
                plan_capture_reader=plan_capture_reader, claim_selection={str(artifact.artifact_id): claims}):
                continue
            facts = [fact for fact in source["facts"] if fact["claim_id"] in claims]
            option.update(eligible=True, reason="", summary="\n".join(fact["summary"] for fact in facts))
            available.append((turn, owner, artifact, claims, facts))
        except (KeyError, TypeError, ValueError):
            continue
    selected_ids = {str(value) for value in included_turn_ids} if included_turn_ids is not None else None
    target_id = available[-1][1].target_id if available else None
    if selected_ids is not None:
        requested = [row for row in available if str(row[0].turn_id) in selected_ids]
        if requested:
            target_id = requested[-1][1].target_id
        if (not selected_ids or len(selected_ids) != len(included_turn_ids)
            or len(requested) != len(selected_ids) or any(row[1].target_id != target_id for row in requested)):
            raise rejected_closure()
    selected = [row for row in available if row[1].target_id == target_id
        and (selected_ids is None or str(row[0].turn_id) in selected_ids)]
    target = await uow.targets.get_scoped(target_id=target_id, domain_id=domain_id) if target_id else None
    if not selected or target is None or target.status != "ENABLED":
        reasons.append({"code": "INVESTIGATION_RECORD_MISSING", "message": "当前选择没有可追溯的有效判断或目标已停用"})
    for option in options:
        option["selected"] = any(str(row[0].turn_id) == option["turn_id"] for row in selected)
    public = {"eligible": not reasons, "reasons": reasons, "conversation_id": str(conversation_id),
        "display_name": conversation.title or "诊断案例候选", "included_turn_ids": [str(row[0].turn_id) for row in selected],
        "turn_candidates": options, "symptoms": [], "root_cause_summary": "", "root_cause_grade": "INCONCLUSIVE",
        "environment": {}, "evidence_candidates": [], "existing_asset_id": None, "source_fingerprint": None, "_context": None}
    if reasons:
        return public
    refs, candidates, claim_refs = [], {}, []
    for turn, owner, artifact, claims, facts in selected:
        ref = closure_reference(turn=turn, run=owner, artifact=artifact, roles=("INVESTIGATION",), claim_ids=claims)
        refs.append(ref)
        claim_refs.extend(closure_claim_refs(artifact, claims))
        links = await uow.turns.list_run_links(turn_id=turn.turn_id)
        for link in links:
            linked_run = await uow.runs.get_run_scoped(ops_run_id=link.ops_run_id, domain_id=domain_id)
            if linked_run is None or linked_run.target_id != target_id:
                continue
            linked_artifacts = await uow.runs.list_artifacts(ops_run_id=linked_run.ops_run_id, limit=513)
            if len(linked_artifacts) > 512:
                raise rejected_closure()
            for item in linked_artifacts:
                if item.schema_version not in {"AIOPS_TURN_RESULT.v1", "DBA_TOOL_RESULT.v1", "DIAGNOSIS_EVIDENCE_COLLECTION.v1",
                    "EXECUTION_RESULT.v1", "USER_PROVIDED_ACTION_RESULT.v1", "ACTION_VERIFICATION.v1", "COMPARISON_RESULT.v1"}:
                    continue
                if item.content_hash != canonical_hash(item.payload_json):
                    continue
                candidates[str(item.artifact_id)] = (turn, linked_run, item)
                if len(candidates) > 512:
                    raise rejected_closure()
        candidates[str(artifact.artifact_id)] = (turn, owner, artifact)
    environment = {"database_type": target.db_type, "database_version": target.version_code,
        "platform": target.environment, "topology": (target.capabilities_json or {}).get("topology")}
    fingerprint = canonical_hash({"conversation_id": str(conversation_id), "actor_id": str(actor_id),
        "target_id": str(target_id), "target_version": target.row_version,
        "turns": [(str(turn.turn_id), turn.status) for turn in turns],
        "roots": [ref.model_dump(mode="json") for ref in refs],
        "evidence": sorted((key, item[2].content_hash) for key, item in candidates.items())})
    public.update(symptoms=list(dict.fromkeys(fact["summary"] for row in selected for fact in row[4])),
        root_cause_summary="\n".join(f"第 {row[0].turn_no} 轮：" + "；".join(fact["summary"] for fact in row[4]) for row in selected),
        environment=environment, source_fingerprint=fingerprint,
        evidence_candidates=[{"artifact_id": key, "turn_id": str(row[0].turn_id), "turn_no": row[0].turn_no,
            "schema_version": row[2].schema_version, "label": f"第 {row[0].turn_no} 轮 · {row[2].schema_version}",
            "role": closure_evidence_role(row[2].schema_version), "created_at": row[2].created_at.isoformat()}
            for key, row in candidates.items()],
        _context={"run_id": selected[-1][1].ops_run_id, "target_id": target_id, "target_version": target.row_version,
            "root_ops_task_id": selected[-1][2].ops_task_id, "source_refs": refs, "claim_refs": claim_refs,
            "candidates": {key: closure_reference(turn=row[0], run=row[1], artifact=row[2], roles=("BEFORE",))
                for key, row in candidates.items()},
            "environment": environment, "security_level": max(ref.security_level for ref in refs)})
    return public


async def close_conversation_case(service, *, domain_id, conversation_id, command, context,
    entity_scope, kc_scope, model_catalog, model_scope, technical_name_field="model_name"):
    """两平台共用收口事务：先冻结精确外部版本，再审批和重验；人工反馈不升级信任。"""
    from aiops_agent.application.case_semantic_redaction import redact_case_for_index
    from aiops_agent.application.operations_knowledge import (
        CASE_COLLECTION, _asset_view, _version_view, _diagnosis_case_markdown, _normalize,
        _validate_closure_evidence,
    )
    from aiops_agent.application.knowledge_withdrawal import queue_case_withdrawals
    from aiops_agent.contracts.knowledge import DiagnosisCaseProfile
    from aiops_agent.entities import (OpsArtifactEntity, OperationsKnowledgeAssetEntity,
        OperationsKnowledgeVersionEntity, OperationsKnowledgeSourceEntity,
        OperationsKnowledgeScopeEntity, OperationsKnowledgeIndexEntity, OperationsKnowledgeReviewEntity)
    from platform_core.identity import uuid7
    actor = service._actor(context)
    preview = await service._conversation_case_context(domain_id=domain_id, conversation_id=conversation_id,
        actor_id=actor, included_turn_ids=command.included_turn_ids)
    internal = preview.get("_context")
    if not preview["eligible"] or internal is None or preview["source_fingerprint"] != command.source_fingerprint:
        raise rejected_closure()
    request_hash = canonical_hash(command.model_dump(mode="json"))
    async with service._uow_factory() as uow:
        existing = await uow.operations_knowledge.find_case_by_external_id(domain_id=domain_id, external_id=str(conversation_id))
        if existing and existing[1].profile_json.get("source", {}).get("request_hash") == request_hash:
            return await service._validated_case_replay(uow=uow, domain_id=domain_id, case=existing)
    candidates = internal["candidates"]
    action_refs, before, after = _validate_closure_evidence(command=command,
        allowed_ids={UUID(key) for key in candidates}, allowed_action_ids={UUID(key)
            for key, ref in candidates.items() if closure_evidence_role(ref.schema_version) == "ACTION"})
    refs = {str(ref.artifact_id): ref for ref in internal["source_refs"]}
    for role, ids in (("ACTION", action_refs), ("BEFORE", sorted(before)), ("AFTER", sorted(after))):
        for identity in ids:
            ref = refs.get(identity)
            refs[identity] = (ref.model_copy(update={"roles": (*ref.roles, role)}) if ref
                else candidates[identity].model_copy(update={"roles": (role,)}))
    source_refs = tuple(refs.values())
    security_level = max(ref.security_level for ref in source_refs)
    symptoms = tuple(dict.fromkeys(value.strip() for value in command.symptoms if value.strip()))
    actions = tuple(dict.fromkeys(value.strip() for value in command.actions if value.strip()))
    if not command.display_name.strip() or not command.root_cause_summary.strip() or not symptoms:
        raise rejected_closure()
    profile = DiagnosisCaseProfile(
        case_kind="NEGATIVE_EXPERIENCE" if actions and action_refs and command.verification_result in {"ADVERSE", "NOT_ACHIEVED"} else "DIAGNOSTIC_REFERENCE",
        problem_signature={"problem_class": "INCIDENT", "normalized_symptoms": list(symptoms)},
        environment=internal["environment"],
        root_cause={"summary": command.root_cause_summary.strip(), "confidence": "UNKNOWN",
            "effective_level": "INCONCLUSIVE", "trust_level": "USER_PROVIDED",
            "evidence_refs": [str(ref.artifact_id) for ref in source_refs if "INVESTIGATION" in ref.roles]},
        actions=tuple({"summary": summary, "source_kind": "USER_PROVIDED", "evidence_refs": action_refs} for summary in actions),
        verification={"result": command.verification_result, "method": "USER_REPORTED",
            "before_refs": sorted(before), "after_refs": sorted(after), "guardrail_refs": []},
        applicability=tuple(value.strip() for value in command.applicability if value.strip()),
        contraindications=tuple(value.strip() for value in command.contraindications if value.strip()),
        source={"conversation_id": str(conversation_id), "turn_ids": preview["included_turn_ids"],
            "run_ids": list(dict.fromkeys(str(ref.run_id) for ref in source_refs)),
            "claim_refs": internal["claim_refs"], "source_fingerprint": command.source_fingerprint,
            "request_hash": request_hash},
    )
    closed_at = datetime.now(UTC)
    closure_id = uuid7()
    def freeze(value):
        closure = DiagnosisClosure(conversation_id=conversation_id,
            source_turn_ids=tuple(UUID(value) for value in preview["included_turn_ids"]),
            source_fingerprint=command.source_fingerprint, source_refs=source_refs,
            target_id=internal["target_id"], target_version=internal["target_version"],
            closed_by=str(actor), closed_at=closed_at, profile=value)
        payload = closure.model_dump(mode="json")
        return SimpleNamespace(artifact_id=closure_id, ops_run_id=internal["run_id"],
            schema_version=closure.schema_version, payload_json=payload,
            content_hash=canonical_hash(payload), security_level=security_level)
    async def source_preflight():
        current = await service._conversation_case_context(domain_id=domain_id, conversation_id=conversation_id,
            actor_id=actor, included_turn_ids=command.included_turn_ids)
        if not current["eligible"] or current["source_fingerprint"] != command.source_fingerprint:
            raise rejected_closure()
        async with service._uow_factory() as uow:
            run = await uow.runs.get_run_scoped(ops_run_id=internal["run_id"], domain_id=domain_id)
            if run is None or not await validate_closure_sources(uow=uow, domain_id=domain_id, run=run,
                artifact=freeze(profile), plan_capture_reader=getattr(service, "_plan_capture_reader", None)):
                raise rejected_closure()
    await source_preflight()
    collection = await service._collection(**kc_scope, name=CASE_COLLECTION, context=context)
    title, profile = await redact_case_for_index(title=command.display_name.strip(), profile=profile, collection=collection,
        structured_model=service._structured_model, prompt_registry=service._prompt_registry,
        model_catalog=model_catalog, model_scope=model_scope, technical_name_field=technical_name_field,
        source_preflight=source_preflight)
    await source_preflight()
    frozen = freeze(profile)
    markdown = _diagnosis_case_markdown(title=title, profile=profile)
    response = await service._kc.ingest_user_file(**kc_scope,
        collection_id=UUID(str(collection["collection_id"])), file_name=f"diagnosis-case-{conversation_id}.md",
        display_name=title, media_type="text/markdown", body=markdown,
        content_sha256=hashlib.sha256(markdown).hexdigest(),
        idempotency_key=f"diagnosis-closure:{conversation_id}:{request_hash}", auth_context=context)
    item = next(iter((response.payload or {}).get("items", [])), None)
    if not item or item.get("status") == "REJECTED":
        raise AIOpsApplicationError(code="AIOPS_KNOWLEDGE_EXTRACTION_FAILED", message="诊断案例候选索引未被受理", status_code=502, retryable=True)
    revision_id = UUID(str(item["bundle_revision_id"]))
    replay = None
    async with service._uow_factory() as uow:
        # 短事务锁定会话，外部审批不跨事务。每个已受理 Revision 都必须有持久化登记。
        await uow.conversations.get_conversation(domain_id=domain_id, conversation_id=conversation_id, lock=True)
        latest = await uow.operations_knowledge.find_case_by_external_id(domain_id=domain_id, external_id=str(conversation_id))
        if latest and latest[1].profile_json.get("source", {}).get("request_hash") == request_hash:
            relations = await uow.operations_knowledge.get_version_relations(asset_version_id=latest[1].asset_version_id)
            if any(str(index.bundle_revision_id) == str(revision_id) for index in relations.get("indexes", ())):
                replay = await service._validated_case_replay(uow=uow, domain_id=domain_id, case=latest)
                return replay
            replay = latest
        asset = latest[0] if latest else OperationsKnowledgeAssetEntity(**entity_scope,
            domain_id=domain_id, asset_id=uuid7(), asset_kind="DIAGNOSIS_CASE", display_name=title,
            status="PROCESSING", security_level=security_level, created_by=actor, updated_by=actor)
        versions = await uow.operations_knowledge.list_versions(domain_id=domain_id, asset_id=asset.asset_id) if latest else []
        version_id = uuid7()
        version = OperationsKnowledgeVersionEntity(**entity_scope, **({"domain_id": domain_id} if entity_scope else {}),
            asset_version_id=version_id, asset_id=asset.asset_id,
            version_no=max((row.version_no for row in versions), default=0) + 1,
            status="REVIEW_REQUIRED" if replay else "PROCESSING", source_hash=frozen.content_hash,
            profile_schema_version=profile.schema_version, profile_json=profile.model_dump(mode="json"),
            extraction_warnings_json=[], created_by=actor)
        scope = {**entity_scope, **({"domain_id": domain_id} if entity_scope else {})}
        artifact_bytes = json.dumps(frozen.payload_json, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        await uow.runs.add_artifact(OpsArtifactEntity(artifact_id=closure_id, ops_run_id=internal["run_id"],
            ops_task_id=internal["root_ops_task_id"], artifact_key=f"diagnosis-closure:{conversation_id}:{frozen.content_hash}",
            artifact_type="DIAGNOSIS_CLOSURE", schema_version=frozen.schema_version, payload_json=frozen.payload_json,
            content_hash=frozen.content_hash, byte_size=len(artifact_bytes),
            provenance_json={"producer": "aiops.diagnosis-closure", "source_fingerprint": command.source_fingerprint},
            trust_level="USER_PROVIDED", security_level=security_level))
        source = OperationsKnowledgeSourceEntity(**scope, asset_version_id=version_id, source_kind="DIAGNOSIS_CLOSURE",
            source_run_id=internal["run_id"], source_artifact_id=closure_id,
            source_external_id=str(conversation_id) if not replay else f"duplicate:{conversation_id}:{revision_id}",
            content_hash=frozen.content_hash, source_locator_json={"turn_ids": preview["included_turn_ids"], "request_hash": request_hash})
        scopes = [OperationsKnowledgeScopeEntity(**scope, asset_version_id=version_id, scope_kind=kind,
            scope_value=str(value), normalized_value=_normalize(str(value)), source_kind="DIAGNOSIS_CLOSURE",
            source_locator_json={"conversation_id": str(conversation_id)}) for kind, value in (
                ("TARGET_ID", internal["target_id"]), ("DATABASE_TYPE", internal["environment"].get("database_type")),
                ("DATABASE_VERSION", internal["environment"].get("database_version")),
                ("PLATFORM", internal["environment"].get("platform")), ("PROBLEM_CLASS", "INCIDENT"),
                ("TOPOLOGY", internal["environment"].get("topology"))) if value]
        index = OperationsKnowledgeIndexEntity(**scope, asset_version_id=version_id,
            collection_id=UUID(str(collection["collection_id"])), bundle_id=UUID(str(item["bundle_id"])),
            bundle_revision_id=revision_id, index_status="PENDING_APPROVAL")
        asset.security_level = max(int(asset.security_level), security_level)
        asset.updated_by = actor
        if latest:
            asset = await uow.operations_knowledge.get_asset(domain_id=domain_id, asset_id=asset.asset_id, for_update=True)
            asset.security_level = max(int(asset.security_level), security_level)
            await uow.operations_knowledge.add_version(version=version, scopes=scopes, sources=[source], indexes=[index])
        else:
            await uow.operations_knowledge.add_asset_version(asset=asset, version=version, scopes=scopes, sources=[source], indexes=[index])
        await uow.operations_knowledge.add_review(OperationsKnowledgeReviewEntity(**scope,
            asset_version_id=version_id, decision="CLOSE_DIAGNOSIS", reviewer_id=actor,
            comment_text="按所选轮次的有效判断生成人工收口候选，不自动发布或授予已验证等级。",
            before_status="PROCESSING", after_status=version.status))
        if replay:
            await queue_case_withdrawals(uow=uow, asset=asset, version=version)
        await uow.commit()
    if replay:
        # 并发受理了不同 Revision：保留并撤回多余确切版本，不能丢弃外部对象。
        async with service._uow_factory() as uow:
            return await service._validated_case_replay(uow=uow, domain_id=domain_id, case=replay)
    if not await service._case_index_source_fence(domain_id=domain_id, version_id=version_id, actor=actor):
        raise rejected_closure()
    await service._kc.review_user_intake(**kc_scope, collection_id=UUID(str(collection["collection_id"])),
        bundle_revision_id=revision_id, decision="APPROVE", comment="诊断收口候选进入索引，仍须业务审核后发布", auth_context=context)
    if not await service._case_index_source_fence(domain_id=domain_id, version_id=version_id, actor=actor, indexed=True):
        raise rejected_closure()
    return {"asset": _asset_view(asset), "version": _version_view(version), "replayed": False}
