from enum import IntEnum, Enum

class Status(IntEnum):
    """状态枚举"""
    ENABLED = 1
    DISABLED = 0

class ParserEngine(str, Enum):
    """解析器引擎枚举"""
    TEXT = "text"
    SQL = "sql"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"

class DbType(str, Enum):
    """Database type enumeration."""
    ORACLE = "oracle"
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"

class IntentType(str, Enum):
    # --- 1. 快捷响应轨 (Direct Response) ---
    CHITCHAT = "chitchat"           # 问候、情感交流
    OFF_TOPIC = "off_topic"         # 拒答、敏感词拦截
    SYSTEM_CMD = "system_command"   # 系统元指令 (清空记忆、设置参数)

    # --- 2. 复杂任务轨 (Planning Required) ---
    # 将原本的 BUSINESS 拆分为：
    KNOWLEDGE_QUERY = "knowledge_query" # 问文：基于知识库的检索 (RAG)
    DATA_ANALYSIS = "data_analysis"     # 问数：涉及 SQL、图表、计算 (Text2SQL)
    TASK_EXECUTION = "task_execution"   # 执行：如“生成报告”、“导出文件”、“发送邮件”
    
    # --- 3. 混合/模糊轨 ---
    COMPLEX_HYBRID = "complex_hybrid"   # 综合：既要问数又要问文，或需要多步推理
    AMBIGUOUS = "ambiguous"             # 模糊：信息不足，需要 Agent 反问（Clarification）

class PacketType(str, Enum):
    METADATA = "metadata"      # 元数据，包含会话 ID、消息 ID 等
    THOUGHT = "thought"       # 思考流
    ANSWER = "answer"         # 最终回复（给用户看）
    SQL_RESULTS = "sql_results" # 结构化 SQL 结果
    DOC_RESULTS = "doc_results" # 文档检索结果
    GRAPH_RESULTS = "graph_results" # 图谱检索结果
    CALL = "call"             # 工具调用状态
    ECHARTS = "echarts"       # 图表数据展示
    ERROR = "error"           # 错误信息
    DONE = "done"             # 结束信号
    WARNING = "warning"       # 警告信息（运维Agent使用）
    REQUIRE_APPROVAL = "require_approval"  # 需要人工审批（运维Agent高危操作门禁）

class AgentCategory(IntEnum):
    """Agent category enumeration."""
    BUSINESS = 1    # 通用业务智能体
    OPS = 2         # 运维智能体

class ChunkType(str, Enum):
    """Knowledge chunk type enumeration."""
    TEXT = "text"
    TABLE = "table"
    PICTURE = "picture"
    HEADING = "heading"

class FeedbackType(IntEnum):
    """Feedback type enumeration."""
    NEUTRAL = 0
    POSITIVE = 1
    NEGATIVE = -1

class FileStatus(IntEnum):
    """File status enumeration."""
    UPLOADED = 1
    PENDING_APPROVE = 2
    APPROVED = 3
    REJECTED = 4
    PARSING = 5
    PARSED = 6
    PARSE_FAILED = 7
    ARCHIVED = 8

class ProcessPriority(IntEnum):
    """Process priority enumeration."""
    HIGH = 3
    MEDIUM = 2
    LOW = 1

class SecurityLevel(IntEnum):
    """Security level enumeration."""  # Fixed incorrect comment from original
    LOW = 1
    MEDIUM = 2
    HIGH = 3

class KbCategory(IntEnum):
    """Knowledge base category enumeration."""
    KBOT = 1
    IMAGE_SEARCH = 2
    GEN_REPORT = 3
    TRANSLATE = 4
    SUMMARY = 5

class KbStatus(IntEnum):
    """Knowledge base status enumeration."""
    DISABLED = 0
    ENABLED = 1
    ARCHIVED = 2

class ModelCategory(IntEnum):
    """Model category enumeration."""
    LLM = 1
    TXT_EMBEDDING = 2
    IMG_EMBEDDING = 3
    RERANKER = 4
    VLM = 5

class PromptCategory(IntEnum):
    """Prompt category enumeration."""
    SYSTEM_PROMPT = 1
    PROMPT_TEMPLATE = 2
    AGENT_PROMPT = 3

class SplitStrategy(IntEnum):
    """Split strategy enumeration."""
    FIXED_SIZE = 1
    DOC_STRUCTURE = 2
    PAGE = 3
    SEMANTIC = 4
    ROW = 5

class AgentStatus(IntEnum):
    """Agent status enumeration."""
    DISABLED = 0
    ENABLED = 1
    ARCHIVED = 2

class MCPToolType(Enum):
    """MCP tool type enumeration."""
    KB_SEARCH = "kb_search"
    FUNCTION_CALL = "function_call"
    INTERNET_SEARCH = "internet_search"
    AGENT_CALL = "agent_call"
    CODE_EXECUTION = "code_execution"
    
class KBSearchType(IntEnum):
    """Knowledge base search type enumeration."""
    VECTOR = 1
    FULLTEXT = 2
    SUMMARY = 3
    GRAPH = 4

class AccessorType(IntEnum):
    """Accessor type enumeration."""
    USER = 1
    SERVICE = 2

class FileCategory(IntEnum):
    """File category enumeration."""
    TEXT = 1
    IMAGE = 2
    AUDIO = 3
    VIDEO = 4
    OTHER = 5

class EmbeddingProvider(str, Enum):
    """Supported embedding service provider enumeration."""
    LOCAL_BGE = "local_bge"
    LOCAL_QWEN = "local_qwen"
    API_QWEN = "api_qwen"
    CHATGPT = "chatgpt"
    OCI = "oci"

class LLMProvider(str, Enum):
    """Supported LLM provider enumeration."""
    API_DEEPSEEK = "api_deepseek"
    API_QWEN = "api_qwen"
    CHATGPT = "chatgpt"
    OCI = "oci"

class RerankerProvider(str, Enum):
    """Supported reranker model enumeration."""
    LOCAL_BGE = "local_bge"
    LOCAL_QWEN = "local_qwen"
    API_QWEN = "api_qwen"

class VLMProvider(str, Enum):
    """Supported VLM provider enumeration."""
    API_QWEN = "api_qwen"
    CHATGPT = "chatgpt"

# Service type enumeration
class ServiceType(str, Enum):
    INTERNAL = "internal"    # Internal service
    EXTERNAL = "external"    # External service
    THIRD_PARTY = "third_party"  # Third-party service

# API Key status enumeration
class APIKeyStatus(str, Enum):
    ACTIVE = "active"
    REVOKED = "revoked"
    EXPIRED = "expired"
    SUSPENDED = "suspended"

# User Token status enumeration
class UserTokenStatus(str, Enum):
    ACTIVE = "active"
    REVOKED = "revoked"
    EXPIRED = "expired"
    LOGGED_OUT = "logged_out"