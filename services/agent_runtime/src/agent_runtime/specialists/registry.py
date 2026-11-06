"""内置 Skill 的显式 Manifest 与实现注册。"""

from agent_runtime.domain.planning import ExecutionMode
from agent_runtime.domain.skills import (
    ArtifactDeclaration,
    DataClassification,
    SkillManifest,
    SkillRegistry,
)

from agent_runtime.composition.knowledge import RETRIEVAL_MANIFEST as KNOWLEDGE_RETRIEVAL_MANIFEST
from agent_runtime.specialists.document.app_knowledge_skill import AppKnowledgeRetrievalSkill
from agent_runtime.specialists.response_composer.app_knowledge_skill import AppKnowledgeComposerSkill
from .conversation import ContextRewriteSkill
from .conversation_response import ConversationResponseSkill
from .data_query import (
    DataQuerySkill,
    MCPDataQueryExecutor,
    SemanticDataQueryExecutor,
)
from .visualization import ChartSkill
from .hybrid import DataConstraintExtractSkill, DocumentScopeExtractSkill
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

DATA_QUERY_MANIFEST = SkillManifest(
    skill_id="data-query",
    version="1.0.0",
    owner="agent-runtime",
    specialist="data_query",
    description="按冻结 Agent 配置调用 MCP 或语义问数 Provider",
    input_schema="DataQueryInput.v1",
    output_artifacts=(
        ArtifactDeclaration(
            artifact_type="QUERY_RESULT",
            schema_version="QUERY_RESULT.v1",
        ),
    ),
    permissions=(),
    execution_mode=ExecutionMode.READ_ONLY,
    idempotent=True,
    timeout_seconds=180,
    max_retries=2,
    data_classification=DataClassification.CONFIDENTIAL,
    external_dependencies=("selectai_aireport", "data_query_api"),
)

CHART_MANIFEST = SkillManifest(
    skill_id="chart",
    version="1.0.0",
    owner="agent-runtime",
    specialist="visualization",
    description="将可信结构化数据编译为跨 App 可渲染的语义图表",
    input_schema="ChartInput.v1",
    output_artifacts=(
        ArtifactDeclaration(
            artifact_type="CHART_SPEC",
            schema_version="CHART_SPEC.v1",
        ),
    ),
    permissions=(),
    execution_mode=ExecutionMode.READ_ONLY,
    idempotent=True,
    timeout_seconds=30,
    max_retries=0,
    data_classification=DataClassification.CONFIDENTIAL,
    external_dependencies=(),
)

DATA_CONSTRAINT_EXTRACT_MANIFEST = SkillManifest(
    skill_id="data-constraint-extract",
    version="1.0.0",
    owner="agent-runtime",
    specialist="hybrid",
    description="读取 App 授权正文后提取受限业务问数约束",
    input_schema="knowledge.app_retrieval.v1",
    output_artifacts=(ArtifactDeclaration(
        artifact_type="DATA_QUERY_CONSTRAINTS",
        schema_version="DataQueryConstraints.v1",
    ),),
    permissions=(),
    execution_mode=ExecutionMode.READ_ONLY,
    idempotent=True,
    timeout_seconds=90,
    max_retries=1,
    data_classification=DataClassification.CONFIDENTIAL,
    external_dependencies=("llm_service", "prompt_registry"),
)

DOCUMENT_SCOPE_EXTRACT_MANIFEST = SkillManifest(
    skill_id="document-scope-extract",
    version="1.0.0",
    owner="agent-runtime",
    specialist="hybrid",
    description="从结构化查询结果提取受限文档检索范围",
    input_schema="QUERY_RESULT.v1",
    output_artifacts=(ArtifactDeclaration(
        artifact_type="DOCUMENT_SCOPE",
        schema_version="DocumentScope.v1",
    ),),
    permissions=(),
    execution_mode=ExecutionMode.READ_ONLY,
    idempotent=True,
    timeout_seconds=90,
    max_retries=1,
    data_classification=DataClassification.CONFIDENTIAL,
    external_dependencies=("llm_service", "prompt_registry"),
)


def register_builtin_skills(
    registry: SkillRegistry,
    *,
    knowledge_integration,
    model_client,
    prompt_resolver,
    service_name: str,
    mcp_data_client=None,
    data_query_client=None,
) -> SkillRegistry:
    """固定注册，不扫描目录、不动态导入用户代码。"""
    mcp_executor = MCPDataQueryExecutor(client=mcp_data_client)
    semantic_executor = SemanticDataQueryExecutor(
        client=data_query_client,
        model_client=model_client,
        prompt_resolver=prompt_resolver,
    )
    registry.register(
        CONTEXT_REWRITE_MANIFEST,
        ContextRewriteSkill(
            model_client=model_client,
            prompt_resolver=prompt_resolver,
        ),
    )
    registry.register(
        KNOWLEDGE_RETRIEVAL_MANIFEST,
        AppKnowledgeRetrievalSkill(clients=knowledge_integration.clients,
            context_factory=knowledge_integration.auth_context, query_factory=knowledge_integration.query),
    )
    registry.register(
        RESPONSE_COMPOSER_MANIFEST,
        ResponseComposerSkill(
            model_client=model_client,
            prompt_resolver=prompt_resolver,
            app_knowledge=AppKnowledgeComposerSkill(composer_factory=knowledge_integration.composer,
                content_port_factory=knowledge_integration.content_port),
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
        DATA_QUERY_MANIFEST,
        DataQuerySkill(
            mcp_executor=mcp_executor,
            semantic_executor=semantic_executor,
        ),
    )
    registry.register(
        CHART_MANIFEST,
        ChartSkill(),
    )
    registry.register(
        DATA_CONSTRAINT_EXTRACT_MANIFEST,
        DataConstraintExtractSkill(
            knowledge_integration=knowledge_integration,
            model_client=model_client,
            prompt_resolver=prompt_resolver,
        ),
    )
    registry.register(
        DOCUMENT_SCOPE_EXTRACT_MANIFEST,
        DocumentScopeExtractSkill(
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
    registry.register(DATA_QUERY_MANIFEST, None)
    registry.register(CHART_MANIFEST, None)
    registry.register(DATA_CONSTRAINT_EXTRACT_MANIFEST, None)
    registry.register(DOCUMENT_SCOPE_EXTRACT_MANIFEST, None)
    return registry
