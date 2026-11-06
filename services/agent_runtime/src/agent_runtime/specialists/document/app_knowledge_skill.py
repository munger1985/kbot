"""新的 App 证据检索 Skill；只从冻结配置选择一个 owner，不回落旧 KC。"""

import inspect
from knowledge_algorithms.contracts.errors import require
from knowledge_algorithms.contracts.identity import AuthorizedScope
from knowledge_algorithms.contracts.query import QueryInput
from knowledge_algorithms.contracts.retrieval import EvidenceSet
from knowledge_algorithms.contracts.runtime import CitationLabel, CitationLabelMap, RetrievalEvidenceArtifact, DelegationClaims
from knowledge_algorithms.contracts.wire import decode, encode
from agent_runtime.runtime import SkillArtifact, SkillResult
from agent_runtime.adapters.knowledge_context import owner_app_id


class AppKnowledgeRetrievalSkill:
    def __init__(self, *, clients, context_factory, query_factory):
        self.clients = dict(clients)
        self.context_factory, self.query_factory = context_factory, query_factory

    async def execute(self, context):
        owner = owner_app_id(context)
        require(owner in self.clients, "冻结 Agent 的知识 owner 未登记")
        configuration = context.config_snapshot.get("app_knowledge")
        require(isinstance(configuration, dict), "运行缺少 App 知识接入快照")
        client = self.clients[owner].for_delegation(configuration["delegation"])
        require(client.owner == owner, "固定知识客户端 owner 不匹配")
        scope = decode(AuthorizedScope, configuration["scope"])
        delegation = decode(DelegationClaims, configuration["delegation"])
        require(delegation.run_ref == str(context.run_id) and delegation.spec_hash == configuration.get("spec_hash")
                and delegation.owner_binding_ref == owner and delegation.scope_hash == scope.scope_hash,
                "冻结 Run/Spec/Scope 与知识委托不一致")
        require(scope.owner_binding_ref == owner and configuration.get("spec_hash"),
                "运行知识范围或冻结 Spec 无效")
        auth_context = self.context_factory(context, scope)
        query = self.query_factory(context)
        if inspect.isawaitable(query):
            query = await query
        require(isinstance(query, QueryInput), "查询规划没有返回正式查询合同")
        raw = await client.retrieve({"scope": encode(scope), "query": encode(query),
            "profile_ref": configuration["profile_ref"]}, auth_context=auth_context)
        evidence = decode(EvidenceSet, raw["evidence_set"])
        require(evidence.owner_binding_ref == owner and evidence.scope_ref == scope.scope_ref
                and evidence.scope_hash == scope.scope_hash and evidence.query_hash == query.query_hash,
                "App 返回了不属于本 Run 的证据")
        primary = {hit.evidence_key for hit in evidence.primary_hits}
        labels = CitationLabelMap(str(context.run_id), tuple(CitationLabel(str(context.run_id),
            "D" + str(index), item.key, "PRIMARY" if item.key in primary else "REFERENCED", item.locators)
            for index, item in enumerate(evidence.evidence, 1)))
        turn_ref = str(configuration["turn_ref"])
        artifact = RetrievalEvidenceArtifact(str(context.run_id), turn_ref, evidence, labels)
        pins = []
        if evidence.evidence:
            for start in range(0, len(evidence.evidence), 128):
                retained = await client.retain({"retrieval_ref": raw["retrieval_ref"],
                    "evidence_keys": [encode(item.key) for item in evidence.evidence[start:start + 128]],
                    "expires_at": configuration["retain_until"]}, auth_context=auth_context)
                pins.extend(retained["pin_ids"])
        return SkillResult(artifact=SkillArtifact(artifact_type="RETRIEVAL_EVIDENCE",
            schema_version="knowledge.app_retrieval.v1",
            payload={"retrieval_ref": raw["retrieval_ref"], "evidence": encode(artifact)},
            provenance={"owner_binding_ref": owner, "scope_hash": scope.scope_hash,
                        "spec_hash": configuration["spec_hash"], "pin_ids": pins}))
