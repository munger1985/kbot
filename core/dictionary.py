from enum import IntEnum, Enum

class Status(IntEnum):
    """Status enumeration."""
    ENABLED = 1
    DISABLED = 0

class YesNoEnum(IntEnum):
    """Yes or No enumeration."""
    YES = 1
    NO = 0

class DbType(IntEnum):
    """Database type enumeration."""
    ORACLE = 1
    ADB = 2
    HEATWAVE = 3
    ELASTICSEARCH = 4
    MILVUS = 5
    FAISS = 6
    PINECONE = 7
    WEAVIATE = 8

class ChunkType(IntEnum):
    """知识块类型枚举"""
    TEXT = 1
    IMAGE = 2
    TABLE = 3
    SUMMARY = 4

class FeedbackType(IntEnum):
    """反馈类型枚举"""
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
    """Process priority enumeration."""
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

class ToolType(IntEnum):
    """Tool type enumeration."""
    KB_SEARCH = 1
    FUNCTION_CALL = 2
    INTERNET_SEARCH = 3
    AGENT_CALL = 4
    CHAT_AI = 5
    CALCULATOR = 6
    CODE_EXECUTION = 7

class MCPToolType(Enum):
    """MCP工具类型枚举"""
    KB_SEARCH = "kb_search"
    FUNCTION_CALL = "function_call"
    INTERNET_SEARCH = "internet_search"
    AGENT_CALL = "agent_call"
    CHAT_AI = "chat_ai"
    CALCULATOR = "calculator"
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
    """支持的嵌入服务提供商枚举"""
    LOCAL_BGE = "local_bge"
    LOCAL_QWEN = "local_qwen"
    API_QWEN = "api_qwen"
    CHATGPT = "chatgpt"
    COHERE = "cohere"

class LLMProvider(str, Enum):
    """支持的LLM提供商枚举"""
    API_DEEPSEEK = "api_deepseek"
    API_QWEN = "api_qwen"
    CHATGPT = "chatgpt"
    OCI = "oci"

class RerankerProvider(str, Enum):
    """支持的 reranker 模型枚举"""
    LOCAL_BGE = "local_bge"
    LOCAL_QWEN = "local_qwen"
    API_QWEN = "api_qwen"
    CHATGPT = "chatgpt"
    COHERE = "cohere"

class VLMProvider(str, Enum):
    """支持的 VLM 提供商枚举"""
    API_QWEN = "api_qwen"
    CHATGPT = "chatgpt"

# 服务类型枚举
class ServiceType(str, Enum):
    INTERNAL = "internal"    # 内部服务
    EXTERNAL = "external"    # 外部服务
    THIRD_PARTY = "third_party"  # 第三方服务

# API Key状态枚举
class APIKeyStatus(str, Enum):
    ACTIVE = "active"
    REVOKED = "revoked"
    EXPIRED = "expired"
    SUSPENDED = "suspended"

# 用户Token状态枚举
class UserTokenStatus(str, Enum):
    ACTIVE = "active"
    REVOKED = "revoked"
    EXPIRED = "expired"
    LOGGED_OUT = "logged_out"