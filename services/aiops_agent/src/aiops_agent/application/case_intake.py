"""会话与报告案例共用同一脱敏、本 App 入库、断点恢复和索引审批流程。"""

import hashlib
from uuid import UUID
from platform_core.identity import uuid7

from aiops_agent.application.errors import AIOpsApplicationError


async def execute_case_intake(service, *, domain_id, prepared, source_validator, replay_finder, register_intake, reject,
    context, entity_scope, kc_scope, model_catalog, model_scope, technical_name_field="model_name",
    actor_id=None, recovery=None, checkpoint=None, record_receipt=None, source_preflight_hook=None):
    """只消费冻结输入；后续无关轮次不阻断，真实反证、原件和权限变化仍重验。"""
    from aiops_agent.application.case_semantic_redaction import redact_case_for_index
    from aiops_agent.application.case_intake_recovery import CaseIntakeRecovery
    from aiops_agent.application.operations_knowledge import (
        _asset_view, _version_view, _diagnosis_case_markdown,
    )
    from aiops_agent.application.knowledge_withdrawal import queue_case_withdrawals
    # 后台以 SERVICE 调 本 App，业务署名来自已核验的任务发起人，不伪造 USER 身份。
    actor = actor_id if actor_id is not None else service._actor(context)
    if prepared.actor_id != str(actor):
        raise reject()
    source_id, request_hash = prepared.source_id, prepared.request_hash
    profile = prepared.profile
    async def source_preflight():
        if source_preflight_hook is not None:
            await source_preflight_hook()
        async with service._uow_factory() as uow:
            await source_validator(uow=uow, domain_id=domain_id, prepared=prepared,
                plan_capture_reader=getattr(service, "_plan_capture_reader", None))
    recovery = CaseIntakeRecovery.model_validate(recovery) if recovery is not None else None
    if recovery is not None:
        recovery.body(prepared=prepared)
    async def replay_receipt(uow, case):
        # 只有确切正文身份属于本次恢复材料时才记录受理；复用其它任务结果不取得其清理权。
        if recovery is None:
            return None
        relations = await uow.operations_knowledge.get_version_relations(asset_version_id=case[1].asset_version_id)
        matches = [row for row in relations["indexes"]
            if row.document_id == recovery.document_id and row.document_version_id == recovery.document_version_id]
        return recovery.model_copy(update={"ingestion_registered": True}) if len(matches) == 1 else None
    if recovery is None or not recovery.ingestion_registered:
        await source_preflight()
        async with service._uow_factory() as uow:
            existing = await replay_finder(uow=uow, domain_id=domain_id, prepared=prepared)
            if existing is not None:
                result = await service._validated_case_replay(uow=uow, domain_id=domain_id, case=existing)
                receipt = await replay_receipt(uow, existing)
        if existing is not None:
            if record_receipt is not None:
                await record_receipt(receipt)
            return result
    already_accepted = recovery is not None and recovery.ingestion_registered
    collection = None if already_accepted else await service._case_configuration(
        domain_id=domain_id, context=context)
    if recovery is not None and not already_accepted and recovery.configuration_ref != collection["profile_ref"]:
        raise reject()
    if recovery is None:
        title, profile = await redact_case_for_index(title=prepared.title, profile=profile, collection=collection,
            structured_model=service._structured_model, prompt_registry=service._prompt_registry,
            model_catalog=model_catalog, model_scope=model_scope, technical_name_field=technical_name_field,
            source_preflight=source_preflight)
        await source_preflight()
        markdown = _diagnosis_case_markdown(title=title, profile=profile)
        recovery = CaseIntakeRecovery(intake_id=prepared.intake_id, request_hash=request_hash,
            source_fingerprint=prepared.source_fingerprint, configuration_ref=collection['profile_ref'],
            document_id=uuid7(), document_version_id=uuid7(),
            title=title, profile=profile, body_sha256=hashlib.sha256(markdown).hexdigest())
        if checkpoint is not None:
            # 外部受理前先持久化正文身份；崩溃后用同一正文和幂等键找回确切 Revision。
            await checkpoint(recovery)
    else:
        recovery = CaseIntakeRecovery.model_validate(recovery)
        markdown = recovery.body(prepared=prepared)
        title, profile = recovery.title, recovery.profile
    if not already_accepted:
        await source_preflight()
    access = service._documents.access(context, actor_id=actor)
    if already_accepted:
        async with service._uow_factory() as uow:
            exact = await uow.operations_knowledge.find_case_by_intake_revision(
                domain_id=domain_id, revision_id=recovery.document_version_id)
            if exact is None:
                raise reject()
            asset, version = exact
            receipt = await replay_receipt(uow, exact)
            if receipt is None:
                raise reject()
        replay, was_existing = None, True
    else:
        staged = await service._documents.stage(file_name=f"diagnosis-case-{source_id}.md",
            media_type="text/markdown", body=markdown, expected_hash=recovery.body_sha256,
            document_id=recovery.document_id, version_id=recovery.document_version_id)
        async with service._uow_factory() as uow:
            asset, version, replay, was_existing = await register_intake(
                uow=uow, domain_id=domain_id, prepared=prepared, recovery=recovery,
                entity_scope=entity_scope, actor=actor, documents=service._documents, access=access, staged=staged)
            if was_existing:
                receipt = await replay_receipt(uow, (asset, version))
            else:
                await uow.commit()
    recovery = recovery.model_copy(update={"ingestion_registered": True})
    if record_receipt is not None:
        await record_receipt(receipt if was_existing else recovery)
    version_id = version.asset_version_id
    if replay:
        # 并发受理了不同 Revision：保留并撤回多余确切版本，不能丢弃外部对象。
        async with service._uow_factory() as uow:
            return await service._validated_case_replay(uow=uow, domain_id=domain_id, case=replay)
    if not await service._case_index_source_fence(domain_id=domain_id, version_id=version_id, actor=actor):
        raise reject()
    if source_preflight_hook is not None:
        try:
            await source_preflight()
        except AIOpsApplicationError as exc:
            if exc.retryable:
                raise
            # 平台权限刚撤销时也留下已受理版本及撤回命令，不能继续批准 本 App。
            async with service._uow_factory() as uow:
                version = await uow.operations_knowledge.get_version(
                    domain_id=domain_id, asset_version_id=version_id, for_update=True)
                asset = await uow.operations_knowledge.get_asset(
                    domain_id=domain_id, asset_id=version.asset_id, for_update=True)
                version.status = "REVIEW_REQUIRED"
                if asset.current_version_id is None:
                    asset.status = "REVIEW_REQUIRED"
                await queue_case_withdrawals(uow=uow, asset=asset, version=version)
                await uow.commit()
            raise
    if not await service._case_index_source_fence(domain_id=domain_id, version_id=version_id, actor=actor, indexed=True):
        raise reject()
    return {"asset": _asset_view(asset), "version": _version_view(version), "replayed": was_existing}
