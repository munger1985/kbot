"""内置 Skill 的显式 Manifest 与实现注册。"""

from agent_runtime.domain.planning import ExecutionMode
from agent_runtime.domain.skills import (
    ArtifactDeclaration,
    DataClassification,
    SkillManifest,
    SkillRegistry,
)

from .document import KnowledgeRetrievalSkill
from .response_composer import ResponseComposerSkill


KNOWLEDGE_RETRIEVAL_MANIFEST = SkillManifest(
    skill_id="knowledge-retrieval",
    version="1.0.0",
    owner="knowledge-core",
    specialist="document",
    description="在授权 Collection 内执行 KC 两阶段检索并生成可引用证据包",
    input_schema="DocumentQueryTask.v1",
    output_artifacts=(
        ArtifactDeclaration(
            artifact_type="CITATION_PACK",
            schema_version="DocumentRetrievalResult.v1",
        ),
    ),
    permissions=(
        "knowledge.discovery.read",
        "knowledge.evidence.read",
    ),
    execution_mode=ExecutionMode.READ_ONLY,
    idempotent=True,
    timeout_seconds=120,
    max_retries=2,
    data_classification=DataClassification.INTERNAL,
    external_dependencies=("knowledge_core_api",),
)


RESPONSE_COMPOSER_MANIFEST = SkillManifest(
    skill_id="response-composer",
    version="1.0.0",
    owner="agent-runtime",
    specialist="response_composer",
    description="仅使用已验证 Artifact 生成最终回答并收敛真实引用",
    input_schema="CompositionInput.v1",
    output_artifacts=(
        ArtifactDeclaration(
            artifact_type="GROUNDED_ANSWER",
            schema_version="GroundedAnswer.v1",
        ),
    ),
    permissions=(),
    execution_mode=ExecutionMode.READ_ONLY,
    idempotent=True,
    timeout_seconds=120,
    max_retries=1,
    data_classification=DataClassification.INTERNAL,
    external_dependencies=("llm_service",),
)


def register_builtin_skills(
    registry: SkillRegistry,
    *,
    knowledge_core_client,
    model_client,
    service_name: str,
) -> SkillRegistry:
    """固定注册，不扫描目录、不动态导入用户代码。"""
    registry.register(
        KNOWLEDGE_RETRIEVAL_MANIFEST,
        KnowledgeRetrievalSkill(
            knowledge_core_client=knowledge_core_client,
            service_name=service_name,
        ),
    )
    registry.register(
        RESPONSE_COMPOSER_MANIFEST,
        ResponseComposerSkill(model_client=model_client),
    )
    return registry


def register_builtin_manifests(registry: SkillRegistry) -> SkillRegistry:
    """API 进程只加载能力声明，不初始化任何下游 Client。"""
    registry.register(KNOWLEDGE_RETRIEVAL_MANIFEST, None)
    registry.register(RESPONSE_COMPOSER_MANIFEST, None)
    return registry
