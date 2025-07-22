from enum import Enum, unique

class Status(int, Enum):
    """Status enumeration."""
    ENABLED = 1
    DISABLED = 0

class YesNoEnum(int, Enum):
    """Yes or No enumeration."""
    YES = 1
    NO = 0

class DbType(int, Enum):
    """Database type enumeration."""
    ORACLE = 1
    ADB = 2
    HEATWAVE = 3
    ELASTICSEARCH = 4
    MILVUS = 5
    FAISS = 6
    PINECONE = 7
    WEAVIATE = 8

class ChunkType(int, Enum):
    """Chunk type enumeration."""
    TEXT = 1
    IMAGE = 2

class FileStatus(int, Enum):
    """File status enumeration."""
    DELETED = -1
    UPLOADED = 1
    PENDING_APPROVE = 2
    APPROVED = 3
    REJECTED = 4
    PARSING = 5
    PARSED = 6
    PARSE_FAILED = 7
    REPARSING = 8
    ARCHIVED = 9

@unique
class ProcessPriority(int, Enum):
    """Process priority enumeration."""
    HIGH = 0
    MEDIUM = 1
    LOW = 2

class SecurityLevel(int, Enum):
    """Process priority enumeration."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3

class KbCategory(int, Enum):
    """Knowledge base category enumeration."""
    KBOT = 1
    IMAGE_SEARCH = 2
    GEN_REPORT = 3
    TRANSLATE = 4
    SUMMARY = 5

class KbStatus(int, Enum):
    """Knowledge base status enumeration."""
    DISABLED = 0
    ENABLED = 1
    ARCHIVED = 2

class ModelCategory(int, Enum):
    """Model category enumeration."""
    LLM = 1
    EMBEDDING = 2
    RERANKER = 3
    VLM = 4

class PromptCategory(int, Enum):
    """Prompt category enumeration."""
    SYSTEM_PROMPT = 1
    PROMPT_TEMPLATE = 2
    AGENT_PROMPT = 3

class ParamType(int, Enum):
    """Parameter type enumeration."""
    SERVICE_URL = 1
    SYSLOGO = 2
    SEARCH = 3
    FEEDBACK = 4
    DATA_PARSE = 5
    GRAPHRAG = 6

class SplitStrategy(int, Enum):
    """Split strategy enumeration."""
    SELF_SPLIT = 1
    BY_DOCSTRUCTURE = 2
    BY_PAGE = 3
    BY_SEMANTIC = 4

class AgentStatus(int, Enum):
    """Agent status enumeration."""
    DISABLED = 0
    ENABLED = 1
    ARCHIVED = 2
