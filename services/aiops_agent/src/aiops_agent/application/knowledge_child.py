"""诊断父轮次下的正式知识子执行；冻结范围、幂等恢复和取消均有明确归属。"""

import asyncio
from loguru import logger
from dataclasses import replace
from datetime import timedelta
import re
import time
from uuid import UUID, NAMESPACE_URL, uuid5

from knowledge_algorithms.contracts.errors import require
from knowledge_algorithms.contracts.identity import fingerprint
from platform_core.contracts.agent import AgentExecutionSpec
from .operations_knowledge_authorization import validate_parent


def child_id(turn_id, purpose):
    return uuid5(NAMESPACE_URL, "aiops-knowledge:" + str(turn_id) + ":" + purpose)


def project_answer(payload, *, run_id, bindings, label_prefix):
    """仅按结构化引用表重编号，不从正文猜来源，也不保留不存在的引用。"""
    references = payload.get("references") or []
    used = payload.get("used_citation_labels") or []
    require(len(set(used)) == len(used)
        and len(references) == len({r["citation_label"] for r in references})
        and set(used) == {r["citation_label"] for r in references},
        "Runtime 知识回答引用闭包不完整")
    by_document = {row["document_version_id"]: row for row in bindings}
    labels, evidence = {}, []
    by_label = {ref["citation_label"]: ref for ref in references}
    for original_label in used:
        ref = by_label[original_label]
        require(ref["reference_type"] == "APP_KNOWLEDGE" and ref["owner_app_id"] == "aiops"
            and ref["run_id"] == str(run_id), "知识子执行返回了其它 App/Run 的引用")
        document = by_document.get(ref["document_version_id"])
        require(document is not None and ref["parse_run_id"] == document["parse_run_id"],
            "知识子执行引用不属于冻结业务选择")
        label = label_prefix + str(len(evidence) + 1)
        labels[ref["citation_label"]] = label
        evidence.append({"label": label, "source_kind": "KNOWLEDGE",
            "source_id": document["asset_version_id"], "asset_version_id": document["asset_version_id"],
            "title": document["title"], "quote": ref.get("quote", ""),
            "record_type": document["asset_kind"], "record_id": document["asset_version_id"],
            "summary": ref.get("quote", ""), "locator": {**ref,
                "asset_id": document["asset_id"], "asset_version_id": document["asset_version_id"],
                "business_context": document.get("business_context", {})}})
    answer = str(payload.get("answer") or "")
    mentioned = set(re.findall(r"\[([A-Z]+[0-9]+)\]", answer))
    require(mentioned == set(labels), "知识子执行正文与结构化引用不一致")
    rendered = re.sub(r"\[([A-Z]+[0-9]+)\]", lambda match: "[" + labels[match[1]] + "]", answer)
    return rendered, tuple(evidence)


class OperationsKnowledgeChild:
    def __init__(self, *, service, client, timeout_seconds=300, poll_seconds=.5,
                 sleep=asyncio.sleep, monotonic=time.monotonic):
        self.service, self.client = service, client
        self.timeout, self.poll, self.sleep, self.monotonic = timeout_seconds, poll_seconds, sleep, monotonic

    async def answer(self, *, context, turn_id, agent_id, agent_version_id, request,
                     target_id=None, label_prefix="K"):
        documents = self.service._documents
        access = documents.access(context)
        run_id, ticket_id = child_id(turn_id, "run"), child_id(turn_id, "ticket")
        async with self.service._uow_factory() as uow:
            await validate_parent(uow, access, turn_id, agent_id, agent_version_id)
            frozen = await uow.knowledge_documents.child_binding(access, ticket_id, now=documents.retrieval.clock())
        if frozen is None:
            result = await self.service.search(
                domain_id=access.domain_id, agent_id=str(agent_id), context=context,
                target_id=target_id, request=request)
            selected = result.get("results") or []
            bindings = [dict(binding, business_context={name: item[name] for name in
                ("authority", "applicability", "reuse_context", "applicability_decision") if name in item})
                for item in selected for binding in item.get("document_bindings", ())]
            unique = {}
            for binding in bindings:
                identity = binding["document_version_id"]
                require(identity not in unique or unique[identity] == binding, "同一文档版本不能对应冲突的业务绑定")
                unique[identity] = binding
            bindings = list(unique.values())
            if not bindings:
                return "没有检索到可以直接支持回答的已发布运维知识。", ()
            require(len(bindings) <= documents.retrieval.limits.max_versions, "知识子执行范围超过预算")
            async with self.service._uow_factory() as uow:
                version = await validate_parent(uow, access, turn_id, agent_id, agent_version_id, lock=True)
                frozen = await uow.knowledge_documents.child_binding(access, ticket_id, now=documents.retrieval.clock())
                if frozen is None:
                    # 角色取自当前诊断 Agent 真实配置；未配置查询规划时显式复用其回答模型。
                    configured = dict(version.models_json or {})
                    require(configured.get("composer_llm"), "诊断 Agent 未配置回答模型")
                    models = {name: configured[name] for name in ("composer_llm", "query_vlm") if configured.get(name)}
                    models["context_llm"] = configured.get("context_llm") or configured.get("router_llm") or configured["composer_llm"]
                    spec = AgentExecutionSpec(schema_version="1.0", owner_app_id="aiops",
                        domain_id=access.domain_id, consumer_agent_id=agent_id,
                        consumer_agent_version_id=agent_version_id, agent_kind="AIOPS_KNOWLEDGE",
                        display_name="运维知识子执行", enabled_capabilities=("document",),
                        models=models, instruction=version.instruction,
                        resource_context={"business_knowledge": [b["business_context"] for b in bindings]},
                        knowledge_binding={"query_ticket": ticket_id, "profile_ref": "default"})
                    delegated = replace(access, run_ref=str(run_id), spec_hash=fingerprint(spec.model_dump(mode="json")))
                    effective = request.model_copy(update={
                        "knowledge_base_ids": tuple(UUID(value) for value in {str(b) for b in request.knowledge_base_ids})})
                    # 策略规范化后再冻结，避免初始请求省略默认范围使当前授权比较永远不相等。
                    from .knowledge_search import _current_knowledge_authorization
                    current = await _current_knowledge_authorization(uow=uow, domain_id=access.domain_id,
                        agent_id=agent_id, target_id=target_id, request=effective)
                    require(not isinstance(current, str), "当前 Agent 知识授权已经撤销")
                    effective, _ = current
                    await uow.knowledge_documents.create_query_ticket(delegated, ticket_id=ticket_id,
                        agent_id=agent_id, target_id=target_id, request=effective, snapshots=bindings,
                        filters={"TAG_ID": tuple(str(v).upper() for v in effective.tag_ids)} if effective.tag_ids else {},
                        parent_turn_id=turn_id, execution_spec=spec.model_dump(mode="json"),
                        expires_at=documents.retrieval.clock() + timedelta(seconds=documents.retrieval.limits.scope_ttl_seconds))
                    frozen = {"execution_spec": spec.model_dump(mode="json"), "snapshots": bindings}
                    await uow.commit()
        spec = AgentExecutionSpec.model_validate(frozen["execution_spec"])
        require(spec.consumer_agent_id == agent_id and spec.consumer_agent_version_id == agent_version_id,
            "不能将已有知识子执行重新绑定其它 Agent")
        started, terminal = False, False
        try:
            started = True  # 请求超时时服务端可能已创建，清理也要尝试查找确定 ID。
            await self.client.create_aiops_knowledge_run(payload={
                "child_run_id": str(run_id), "parent_turn_id": str(turn_id),
                "execution_spec": spec.model_dump(mode="json"), "input": request.query},
                idempotency_key="aiops-knowledge:" + str(turn_id), auth_context=context)
            deadline = self.monotonic() + self.timeout
            while self.monotonic() < deadline:
                async with self.service._uow_factory() as uow:
                    await validate_parent(uow, access, turn_id, agent_id, agent_version_id)
                state = await self.client.get_run(run_id=run_id, auth_context=context)
                if state["status"] in {"FAILED", "CANCELLED", "EXPIRED", "TIMED_OUT"}:
                    terminal = True
                    raise ValueError("知识子执行失败，未生成无来源回答")
                if state["status"] == "COMPLETED":
                    artifact = await self.client.get_result(run_id=run_id, auth_context=context)
                    require(artifact["artifact_type"] == "GROUNDED_ANSWER", "知识子执行没有正式回答产物")
                    payload = dict(artifact["payload"])
                    for ref in payload.get("references", ()):
                        # 下载走 Runtime 精确历史证据读取及 App 当前授权，不调用 Top-K 重新搜索。
                        download = await self.client.open_knowledge_content(
                            run_id=run_id, key=ref["evidence_key"], auth_context=context)
                        try:
                            body = bytearray()
                            async for block in download.blocks:
                                require(len(body) + len(block) <= 8388608, "知识引用正文超出预算")
                                body.extend(block)
                            ref["quote"] = bytes(body).decode("utf-8") if download.mime_type.startswith("text/") else ""
                        finally:
                            await download.close()
                    async with self.service._uow_factory() as uow:
                        await validate_parent(uow, access, turn_id, agent_id, agent_version_id)
                    answer = project_answer(payload, run_id=run_id, bindings=frozen["snapshots"], label_prefix=label_prefix)
                    terminal = True
                    return answer
                await self.sleep(self.poll)
            raise TimeoutError("知识子执行超过本轮等待预算")
        finally:
            if started and not terminal:
                try:
                    await asyncio.shield(self._cancel(run_id, context))
                except Exception as exc:
                    logger.warning("知识子执行取消待恢复：run_id={} error_type={}", run_id, type(exc).__name__)

    async def _cancel(self, run_id, context):
        """取消冲突时重读版本；失败仍由 Runtime 截止时间和证据清理负责回收。"""
        from platform_clients.agent_runtime import AgentRuntimeClientError
        for attempt in range(3):
            try:
                state = await self.client.get_run(run_id=run_id, auth_context=context)
                if state["status"] in {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}:
                    return
                await self.client.cancel_run(run_id=run_id, expected_row_version=state["row_version"],
                    idempotency_key="aiops-knowledge-cancel:" + str(run_id), auth_context=context)
                return
            except AgentRuntimeClientError as exc:
                if exc.status_code == 409:
                    continue
                if exc.status_code == 404:
                    return
                raise
