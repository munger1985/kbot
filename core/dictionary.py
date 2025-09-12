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
    TABLE = 3

class FileStatus(int, Enum):
    """File status enumeration."""
    UPLOADED = 1
    PENDING_APPROVE = 2
    APPROVED = 3
    REJECTED = 4
    PARSING = 5
    PARSED = 6
    PARSE_FAILED = 7
    ARCHIVED = 8

@unique
class ProcessPriority(int, Enum):
    """Process priority enumeration."""
    HIGH = 3
    MEDIUM = 2
    LOW = 1

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

class SplitStrategy(int, Enum):
    """Split strategy enumeration."""
    FIXED_SIZE = 1
    DOC_STRUCTURE = 2
    PAGE = 3
    SEMANTIC = 4
    ROW = 5

class AgentStatus(int, Enum):
    """Agent status enumeration."""
    DISABLED = 0
    ENABLED = 1
    ARCHIVED = 2

class ToolType(int, Enum):
    """Tool type enumeration."""
    KB = 1
    FUNCTIONCALL = 2
    INTERNET = 3
    AGENT = 4
    CHATAI = 5

class KBSearchType(int, Enum):
    """Knowledge base search type enumeration."""
    VECTOR = 1
    FULLTEXT = 2
    SUMMARY = 3
    GRAPH = 4

class AccessorType(int, Enum):
    """Accessor type enumeration."""
    USER = 1
    SERVICE = 2
