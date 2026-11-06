"""App 知识 Runtime 的单一路径组合根；不注册旧 KC 两阶段 Skill。"""

from dataclasses import dataclass
from knowledge_algorithms.contracts.errors import require
from knowledge_algorithms.contracts.runtime import DelegationClaims
from knowledge_algorithms.contracts.wire import encode
from agent_runtime.domain.planning import ExecutionMode
from agent_runtime.domain.skills import ArtifactDeclaration, DataClassification, SkillManifest


RETRIEVAL_MANIFEST = SkillManifest(skill_id="knowledge-retrieval", version="1.0.0",
    owner="agent-runtime", specialist="document", description="仅检索冻结 owner 的授权资料并生成统一多模态证据",
    input_schema="knowledge.query_input.v1",
    output_artifacts=(ArtifactDeclaration(artifact_type="RETRIEVAL_EVIDENCE",
        schema_version="knowledge.app_retrieval.v1"),),
    permissions=("knowledge.retrieve", "knowledge.content.read"),
    execution_mode=ExecutionMode.READ_ONLY, idempotent=True, timeout_seconds=120, max_retries=1,
    data_classification=DataClassification.CONFIDENTIAL,
    external_dependencies=("app_knowledge_api",))

COMPOSER_MANIFEST = SkillManifest(skill_id="response-composer", version="1.0.0",
    owner="agent-runtime", specialist="response_composer", description="实际交付允许的文本/图片并审计引用闭包",
    input_schema="knowledge.app_retrieval.v1",
    output_artifacts=(ArtifactDeclaration(artifact_type="KNOWLEDGE_COMPOSITION",
        schema_version="knowledge.composition.v1"),),
    permissions=("knowledge.content.read",), execution_mode=ExecutionMode.READ_ONLY,
    idempotent=False, timeout_seconds=120, max_retries=0,
    data_classification=DataClassification.CONFIDENTIAL,
    external_dependencies=("app_knowledge_api", "model_serving"))


def register_app_knowledge_skills(registry, *, clients, context_factory, query_factory,
                                  composer_factory, content_port_factory):
    """调用方须传空知识注册槽，不覆盖同版本 Manifest，也不保留新旧检索 fallback。"""
    from agent_runtime.specialists.document.app_knowledge_skill import AppKnowledgeRetrievalSkill
    from agent_runtime.specialists.response_composer.app_knowledge_skill import AppKnowledgeComposerSkill
    registry.register(RETRIEVAL_MANIFEST, AppKnowledgeRetrievalSkill(clients=clients,
        context_factory=context_factory, query_factory=query_factory))
    registry.register(COMPOSER_MANIFEST, AppKnowledgeComposerSkill(composer_factory=composer_factory,
        content_port_factory=content_port_factory))
    return registry


@dataclass(frozen=True, slots=True)
class KnowledgeExecutionBinding:
    """由可信 App Spec 编译器生成，不是公开请求或旧 AgentExecutionSpec。"""
    run_id: str
    spec_hash: str
    owner_binding_ref: str

    def __post_init__(self):
        require(self.run_id and self.spec_hash and self.owner_binding_ref in {"knowledge_retrieval", "aiops"},
                "知识执行绑定无效")


def freeze_knowledge_context(*, trusted_spec, scope, delegation, profile_ref, turn_ref, model_snapshot_ref,
                             payload_budget, retain_until, search_only):
    """只接受 App 通过授权及 Spec 编译形成的上下文；此函数不是公开请求 DTO。"""
    require(isinstance(trusted_spec, KnowledgeExecutionBinding) and isinstance(delegation, DelegationClaims)
            and delegation.run_ref == str(trusted_spec.run_id)
            and delegation.spec_hash == trusted_spec.spec_hash and delegation.scope_hash == scope.scope_hash
            and delegation.owner_binding_ref == trusted_spec.owner_binding_ref == scope.owner_binding_ref,
            "知识上下文与可信执行 Spec 不一致")
    require(type(search_only) is bool and type(retain_until) is int
            and 0 < retain_until <= delegation.expires_at + 2592000, "运行保留/搜索配置无效")
    return {"scope": encode(scope), "delegation": encode(delegation), "spec_hash": trusted_spec.spec_hash,
            "profile_ref": profile_ref, "turn_ref": turn_ref, "model_snapshot_ref": model_snapshot_ref,
            "payload_budget": encode(payload_budget), "retain_until": retain_until, "search_only": search_only}


def build_app_knowledge_registry(**dependencies):
    """供正式 Worker 注入的完整单路径注册表；不自动拼入旧 KC Skill。"""
    from agent_runtime.domain.skills import SkillRegistry
    return register_app_knowledge_skills(SkillRegistry(), **dependencies)


def build_app_knowledge_plan(*, objective, expires_at):
    """生成检索到交付的固定 DAG；纯搜索由同一 Composer 明确 NOT_APPLICABLE。"""
    from agent_runtime.domain.planning import ExecutionKind, PlanDraft, TaskSpec
    retrieval = TaskSpec(task_key="retrieve", task_type="DOCUMENT",
        execution_kind=ExecutionKind.LOCAL_SKILL, specialist="document",
        skill_id=RETRIEVAL_MANIFEST.skill_id, skill_version=RETRIEVAL_MANIFEST.version,
        expected_outputs=("RETRIEVAL_EVIDENCE",), required_scopes=RETRIEVAL_MANIFEST.permissions,
        timeout_seconds=RETRIEVAL_MANIFEST.timeout_seconds, max_retries=RETRIEVAL_MANIFEST.max_retries,
        execution_mode=ExecutionMode.READ_ONLY)
    compose = TaskSpec(task_key="compose", task_type="COMPOSITION",
        execution_kind=ExecutionKind.LOCAL_SKILL, specialist="response_composer",
        skill_id=COMPOSER_MANIFEST.skill_id, skill_version=COMPOSER_MANIFEST.version,
        depends_on=("retrieve",), input_refs=("task_output:retrieve",), expected_outputs=("KNOWLEDGE_COMPOSITION",),
        required_scopes=COMPOSER_MANIFEST.permissions, timeout_seconds=COMPOSER_MANIFEST.timeout_seconds,
        max_retries=0, execution_mode=ExecutionMode.READ_ONLY)
    return PlanDraft(plan_version="app-knowledge/1.0.0", objective=objective,
                     tasks=(retrieval, compose), final_task_key="compose", expires_at=expires_at)
