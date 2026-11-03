"""会话与报告案例共用同一脱敏、KC 入库、断点恢复和索引审批流程。"""

import hashlib
from uuid import UUID

from aiops_agent.application.errors import AIOpsApplicationError


async def execute_case_intake(service, *, domain_id, prepared, source_validator, replay_finder, register_intake, reject,
    context, entity_scope, kc_scope, model_catalog, model_scope, technical_name_field="model_name",
    actor_id=None, recovery=None, checkpoint=None, record_receipt=None, source_preflight_hook=None):
    """只消费冻结输入；后续无关轮次不阻断，真实反证、原件和权限变化仍重验。"""
    from aiops_agent.application.case_semantic_redaction import redact_case_for_index
    from aiops_agent.application.case_intake_recovery import CaseIntakeRecovery
    from aiops_agent.application.operations_knowledge import (
        CASE_COLLECTION, _asset_view, _version_view, _diagnosis_case_markdown,
    )
    from aiops_agent.application.knowledge_withdrawal import queue_case_withdrawals
    # 后台以 SERVICE 调 KC，业务署名来自已核验的任务发起人，不伪造 USER 身份。
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
    if recovery is None or recovery.bundle_revision_id is None:
        await source_preflight()
        async with service._uow_factory() as uow:
            existing = await replay_finder(uow=uow, domain_id=domain_id, prepared=prepared)
            if existing is not None:
                return await service._validated_case_replay(uow=uow, domain_id=domain_id, case=existing)
    already_accepted = recovery is not None and recovery.bundle_revision_id is not None
    if already_accepted:
        # 受理已发生时先恢复 Registry 登记，再由下方围栏暂停／精确撤回。
        # 这一步不再调用模型、上传或批准外部索引。
        collection_id = recovery.collection_id
    else:
        await source_preflight()
        collection = await service._collection(**kc_scope, name=CASE_COLLECTION, context=context)
        collection_id = UUID(str(collection["collection_id"]))
    if recovery is None:
        title, profile = await redact_case_for_index(title=prepared.title, profile=profile, collection=collection,
            structured_model=service._structured_model, prompt_registry=service._prompt_registry,
            model_catalog=model_catalog, model_scope=model_scope, technical_name_field=technical_name_field,
            source_preflight=source_preflight)
        await source_preflight()
        markdown = _diagnosis_case_markdown(title=title, profile=profile)
        recovery = CaseIntakeRecovery(intake_id=prepared.intake_id, request_hash=request_hash,
            source_fingerprint=prepared.source_fingerprint, collection_id=collection_id,
            title=title, profile=profile, body_sha256=hashlib.sha256(markdown).hexdigest())
        if checkpoint is not None:
            # 外部受理前先持久化正文身份；崩溃后用同一正文和幂等键找回确切 Revision。
            await checkpoint(recovery)
    else:
        recovery = CaseIntakeRecovery.model_validate(recovery)
        markdown = recovery.body(prepared=prepared, collection_id=collection_id)
        title, profile = recovery.title, recovery.profile
    if not already_accepted:
        await source_preflight()
    if recovery.bundle_revision_id is None:
        response = await service._kc.ingest_user_file(**kc_scope,
            collection_id=collection_id, file_name=f"diagnosis-case-{source_id}.md",
            display_name=title, media_type="text/markdown", body=markdown,
            content_sha256=recovery.body_sha256,
            idempotency_key=f"case-intake:{prepared.source_kind}:{source_id}:{request_hash}", auth_context=context)
        item = next(iter((response.payload or {}).get("items", [])), None)
        if not item or item.get("status") == "REJECTED":
            raise AIOpsApplicationError(code="AIOPS_KNOWLEDGE_EXTRACTION_FAILED", message="诊断案例候选索引未被受理", status_code=502, retryable=True)
        recovery = recovery.model_copy(update={"bundle_id": UUID(str(item["bundle_id"])),
            "bundle_revision_id": UUID(str(item["bundle_revision_id"]))})
        if record_receipt is not None:
            # 即使刚发生撤权也先登记外部回执，不据此批准索引或完成业务任务。
            await record_receipt(recovery)
    revision_id = recovery.bundle_revision_id
    async with service._uow_factory() as uow:
        asset, version, replay, was_existing = await register_intake(
            uow=uow, domain_id=domain_id, prepared=prepared, recovery=recovery,
            entity_scope=entity_scope, actor=actor)
        if was_existing:
            return await service._validated_case_replay(uow=uow, domain_id=domain_id, case=(asset, version))
        await uow.commit()
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
            # 平台权限刚撤销时也留下已受理版本及撤回命令，不能继续批准 KC。
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
    await service._kc.review_user_intake(**kc_scope, collection_id=collection_id,
        bundle_revision_id=revision_id, decision="APPROVE", comment="案例候选进入索引，仍须业务审核后发布", auth_context=context)
    if not await service._case_index_source_fence(domain_id=domain_id, version_id=version_id, actor=actor, indexed=True):
        raise reject()
    return {"asset": _asset_view(asset), "version": _version_view(version), "replayed": False}
