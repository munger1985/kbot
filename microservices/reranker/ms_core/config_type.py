import humanfriendly
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Literal


class ParserConfig(BaseModel):
    max_workers: int = Field(gt=0)
    check_interval: int = Field(ge=1)  # 至少1秒

class LogConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
    dir: str
    rotation: str  # 如 "100 MB"
    retention: str  # 如 "10 days"

    @field_validator('rotation', 'retention', mode='before')
    @classmethod
    def validate_human_friendly(cls, v: str) -> str:
        try:
            if "MB" in v or "GB" in v:
                humanfriendly.parse_size(v)
            elif "days" in v or "hours" in v:
                humanfriendly.parse_timespan(v)
            return v
        except humanfriendly.InvalidSize as e:
            raise ValueError(f"Invalid format: {v}") from e

class KBotConfig(BaseModel):
    title: str
    description: str
    version: str
    debug: bool
    file_storage: str
    upload_workers: int = Field(ge=1)
    parser: ParserConfig
    log: LogConfig

class LibreConfig(BaseModel):
    host: str
    port: int = Field(gt=0, lt=65536)

class AppConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')  # 禁止额外字段
    
    kbot: KBotConfig
    libre: LibreConfig

    # V2 的根验证器示例
    @field_validator('kbot', mode='after')
    @classmethod
    def validate_kbot(cls, v: KBotConfig) -> KBotConfig:
        if v.debug and v.log.level != "DEBUG":
            raise ValueError("Debug mode requires DEBUG log level")
        return v
    
class OracleConfig(BaseModel):
    host: str
    port: int = Field(gt=0, lt=65536)
    username: str
    password: str
    service_name: str

class RedisConfig(BaseModel):
    host: str
    port: int = Field(gt=0, lt=65536)
    password: str
    max_connections: int = Field(ge=1)
    socket_connect_timeout: int = Field(ge=1)
    socket_timeout: int = Field(ge=1)
    retry_on_timeout: bool
    health_check_interval: int = Field(ge=1)

class SQLAlchemyConfig(BaseModel):
    echo: bool
    pool_size: int = Field(ge=1)
    pool_timeout: int = Field(ge=1) # seconds to wait for a connection
    max_overflow: int = Field(ge=0)
    pool_pre_ping: bool             # test connections for liveness before use
    pool_recycle: int = Field(ge=0) # recycle connections after 1 hour
    pool_use_lifo: bool

class EslogConfig(BaseModel):
    hosts: list[str]
    username: str
    password: str
    index: str
    
class DBConfig(BaseModel):
    oracle: OracleConfig
    redis: RedisConfig
    sqlalchemy: SQLAlchemyConfig
    eslog: EslogConfig

class EmbedConfig(BaseModel):
    service_name: str
    service_version: str
    service_host: str
    service_port: int = Field(gt=0, lt=65536)
    max_tokens: int = Field(ge=1)
    timeout: int = Field(ge=1)
    max_retries: int = Field(ge=0)
    cache_dir: str

class LLMConfig(BaseModel):
    service_name: str
    service_version: str
    service_host: str
    service_port: int = Field(gt=0, lt=65536)
    max_tokens: int = Field(ge=1)
    temperature: float = Field(ge=0.0, le=2.0)
    top_p: float = Field(ge=0.0, le=1.0)
    top_k: int = Field(ge=0)
    timeout: int = Field(ge=1)
    frequency_penalty: float = Field(ge=0.0, le=2.0)
    presence_penalty: float = Field(ge=0.0, le=2.0)

class RerankerConfig(BaseModel):
    service_name: str
    service_version: str
    service_host: str
    service_port: int = Field(gt=0, lt=65536)
    cache_dir: str
    timeout: int = Field(ge=1)

class VLMConfig(BaseModel):
    service_name: str
    service_version: str
    service_host: str
    service_port: int = Field(gt=0, lt=65536)
    timeout: int = Field(ge=1)

class TokenizerConfig(BaseModel):
    custom_dict_path: str
    stop_words_path: str

class PromptConfig(BaseModel):
    image2text: str
    summary: str

class ModelConfig(BaseModel):
    embed: EmbedConfig
    llm: LLMConfig
    reranker: RerankerConfig
    vlm: VLMConfig
    tokenizer: TokenizerConfig
    prompt: PromptConfig

