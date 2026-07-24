"""内置 Skill 的显式 Manifest 与实现注册。"""

from agent_runtime.domain.planning import ExecutionMode
from agent_runtime.domain.skills import (
    ArtifactDeclaration,
    DataClassification,
    SkillManifest,
    SkillRegistry,
)

from .document import KnowledgeRetrievalSkill
from .conversation import ContextRewriteSkill
from .conversation_response import ConversationResponseSkill
from .mcp_data import EChartsSkill, MCPDataQuerySkill
from .response_composer import ResponseComposerSkill


CONTEXT_REWRITE_MANIFEST = SkillManifest(
    skill_id="context-rewrite",
    version="1.0.0",
    owner="agent-runtime",
    specialist="conversation",
    description="根据冻结会话上下文改写独立问题并显式报告歧义",
    input_schema="ConversationContextTask.v1",
    output_artifacts=(
        ArtifactDeclaration(
            artifact_type="CONTEXT_REWRITE",
            schema_version="ContextRewriteOutput.v1",
        ),
    ),
    permissions=(),
    execution_mode=ExecutionMode.READ_ONLY,
    idempotent=True,
    timeout_seconds=60,
    max_retries=1,
    data_classification=DataClassification.INTERNAL,
    external_dependencies=("llm_service", "prompt_registry"),
)


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
    external_dependencies=("knowledge_core_api", "vlm_service"),
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

CONVERSATION_RESPONSE_MANIFEST = SkillManifest(
    skill_id="conversation-response",
    version="1.0.0",
    owner="agent-runtime",
    specialist="conversation",
    description="不调用领域工具，生成通用对话流式回答",
    input_schema="ConversationResponseInput.v1",
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
    external_dependencies=("llm_service", "prompt_registry"),
)

MCP_DATA_QUERY_MANIFEST = SkillManifest(
    skill_id="mcp-data-query",
    version="1.0.0",
    owner="agent-runtime",
    specialist="mcp_data",
    description="通过已配置 Profile 调用 SelectAI/AIReport 问数接口",
    input_schema="MCPDataQueryInput.v1",
    output_artifacts=(
        ArtifactDeclaration(
            artifact_type="QUERY_RESULT",
            schema_version="QueryResult.v1",
        ),
    ),
    permissions=(),
    execution_mode=ExecutionMode.READ_ONLY,
    idempotent=True,
    timeout_seconds=180,
    max_retries=2,
    data_classification=DataClassification.CONFIDENTIAL,
    external_dependencies=("selectai_aireport",),
)

ECHARTS_MANIFEST = SkillManifest(
    skill_id="echarts",
    version="1.0.0",
    owner="agent-runtime",
    specialist="mcp_data",
    description="将 QUERY_RESULT 转换为前端可直接渲染的 ECharts option",
    input_schema="EChartsInput.v1",
    output_artifacts=(
        ArtifactDeclaration(
            artifact_type="ECHARTS_CONFIG",
            schema_version="EChartsResult.v1",
        ),
    ),
    permissions=(),
    execution_mode=ExecutionMode.READ_ONLY,
    idempotent=True,
    timeout_seconds=120,
    max_retries=1,
    data_classification=DataClassification.CONFIDENTIAL,
    external_dependencies=("llm_service", "prompt_registry"),
)


def register_builtin_skills(
    registry: SkillRegistry,
    *,
    knowledge_core_client,
    model_client,
    prompt_resolver,
    service_name: str,
    mcp_data_client=None,
) -> SkillRegistry:
    """固定注册，不扫描目录、不动态导入用户代码。"""
    registry.register(
        CONTEXT_REWRITE_MANIFEST,
        ContextRewriteSkill(
            model_client=model_client,
            prompt_resolver=prompt_resolver,
        ),
    )
    registry.register(
        KNOWLEDGE_RETRIEVAL_MANIFEST,
        KnowledgeRetrievalSkill(
            knowledge_core_client=knowledge_core_client,
            model_client=model_client,
            prompt_resolver=prompt_resolver,
            service_name=service_name,
        ),
    )
    registry.register(
        RESPONSE_COMPOSER_MANIFEST,
        ResponseComposerSkill(
            model_client=model_client,
            prompt_resolver=prompt_resolver,
        ),
    )
    registry.register(
        CONVERSATION_RESPONSE_MANIFEST,
        ConversationResponseSkill(
            model_client=model_client,
            prompt_resolver=prompt_resolver,
        ),
    )
    registry.register(
        MCP_DATA_QUERY_MANIFEST,
        MCPDataQuerySkill(data_client=mcp_data_client),
    )
    registry.register(
        ECHARTS_MANIFEST,
        EChartsSkill(
            model_client=model_client,
            prompt_resolver=prompt_resolver,
        ),
    )
    return registry


def register_builtin_manifests(registry: SkillRegistry) -> SkillRegistry:
    """API 进程只加载能力声明，不初始化任何下游 Client。"""
    registry.register(CONTEXT_REWRITE_MANIFEST, None)
    registry.register(KNOWLEDGE_RETRIEVAL_MANIFEST, None)
    registry.register(RESPONSE_COMPOSER_MANIFEST, None)
    registry.register(CONVERSATION_RESPONSE_MANIFEST, None)
    registry.register(MCP_DATA_QUERY_MANIFEST, None)
    registry.register(ECHARTS_MANIFEST, None)
    return registry
