"""AIOps 查询委托的当前业务授权；不能仅凭同域或 Runtime 身份取得正文。"""

from uuid import UUID

from knowledge_algorithms.contracts.errors import require
from knowledge_algorithms.contracts.identity import fingerprint
from aiops_agent.domain.knowledge_ingestion import utcnow
from aiops_agent.domain.knowledge_retrieval import KnowledgeGrant
from aiops_agent.contracts.knowledge import KnowledgeSearchRequest
from .knowledge_search import _current_knowledge_authorization, _eligible_case


class OperationsKnowledgeAuthorization:
    def __init__(self, *, service, clock=utcnow):
        self.service, self.clock = service, clock

    async def authorize(self, access, resource_ref):
        require(resource_ref.startswith("ops-query:"), "AIOps 知识读取必须使用本 App 签发的查询委托")
        identity = UUID(resource_ref[len("ops-query:"):])
        async with self.service._uow_factory() as uow:
            ticket = await uow.knowledge_documents.query_ticket(access, identity, now=self.clock(), allow_expired=access.historical_read)
            if ticket.get("parent_turn_id"):
                version = await validate_parent(uow, access, UUID(ticket["parent_turn_id"]),
                    UUID(ticket["agent_id"]), UUID(ticket["execution_spec"]["consumer_agent_version_id"]))
                require(fingerprint(ticket["execution_spec"]) == access.spec_hash,
                    "知识查询委托与当前执行规格不一致")
            request = KnowledgeSearchRequest.model_validate(ticket["request"])
            agent_id = UUID(ticket["agent_id"])
            target_id = UUID(ticket["target_id"]) if ticket["target_id"] else None
            current = await _current_knowledge_authorization(uow=uow, domain_id=access.domain_id,
                agent_id=agent_id, target_id=target_id, request=request)
            require(not isinstance(current, str), "Agent 知识策略、绑定或目标权限已撤销")
            effective, base_ids = current
            require(effective == request, "Agent 知识策略已改变，原查询委托不再有效")
            expected = {row["index_ref_id"]: row for row in ticket["snapshots"]}
            rows = await uow.operations_knowledge.list_published_candidates(
                domain_id=access.domain_id, target_id=target_id,
                source_kinds=request.source_kinds, knowledge_base_ids=base_ids,
                max_security_level=request.max_security_level, scope_filters=ticket.get("filters", {}),
                asset_version_ids=tuple(UUID(row["asset_version_id"]) for row in expected.values()),
                limit=None)
            permitted, facts = [], []
            for asset, version, index in rows:
                frozen = expected.get(str(index.index_ref_id))
                if frozen is None:
                    continue
                identity_matches = all(str(getattr(index, name)) == frozen[name] for name in (
                    "document_id", "document_version_id", "ingestion_job_id", "parse_run_id", "index_generation_id"))
                if not identity_matches or not await _eligible_case(self.service, uow=uow,
                    domain_id=access.domain_id, agent_id=agent_id, asset=asset, version=version):
                    continue
                permitted.append(UUID(frozen["document_version_id"]))
                facts.append({"index_ref_id": str(index.index_ref_id),
                    "asset_version_id": str(version.asset_version_id),
                    "profile": version.profile_json, "security_level": int(asset.security_level),
                    "version": int(version.row_version), "asset": int(asset.row_version)})
        revision = fingerprint({"request": ticket["request"], "facts": sorted(facts, key=lambda row: row["index_ref_id"])})
        return KnowledgeGrant(revision, tuple(sorted(set(permitted), key=str)))

async def validate_parent(uow, access, turn_id, agent_id, agent_version_id, *, lock=False):
    """每次读取复核父轮次、会话、当前 Agent 版本，撤权后不得继续交付。"""
    turn = await uow.turns.get_turn(domain_id=access.domain_id, turn_id=turn_id, lock=lock)
    require(turn is not None and str(turn.created_by) == access.subject_ref
        and turn.status not in {"CANCELLED", "FAILED", "EXPIRED", "DELETED"},
        "父诊断轮次已撤销或不属于当前主体")
    conversation = await uow.conversations.get_conversation(
        domain_id=access.domain_id, conversation_id=turn.conversation_id)
    require(conversation is not None and conversation.status == "ACTIVE"
        and str(conversation.created_by) == access.subject_ref
        and conversation.agent_id == agent_id and conversation.agent_version_id == agent_version_id,
        "父会话已删除或 Agent 版本发生变化")
    agent = await uow.agents.get(domain_id=access.domain_id, agent_id=agent_id)
    require(agent is not None and agent.status == "ACTIVE" and agent.current_version_id == agent_version_id,
        "当前 Agent 已停用或切换版本")
    version = await uow.agents.version(agent_id=agent_id, agent_version_id=agent_version_id)
    require(version is not None, "诊断 Agent 版本不存在")
    return version
