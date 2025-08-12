from typing import TypedDict


class AppConfig(TypedDict):
    name: str
    description: str
    version: str
    debug: bool

class DatabaseConfig(TypedDict):
    url: str
    echo: bool
    pool_size: int
    max_overflow: int
    pool_pre_ping: bool
    pool_recycle: int

class RedisConfig(TypedDict):
    url: str
    password: str
    max_connections: int

class LoggerConfig(TypedDict):
    level: str
    dir: str
    rotation: str
    retention: str

class EmbedConfig(TypedDict):
    batch_size: int
    max_tokens: int
    timeout: int
    max_retries: int

class LLMConfig(TypedDict):
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    max_retries: int
    timeout: int

class NlpConfig(TypedDict):
    model_name: str

class KBotConfig(TypedDict):
    file_root_path: str
    parallel_workers: int

class ParserConfig(TypedDict):
    max_workers: int
    check_interval: int

class AppSettings(TypedDict):
    app: AppConfig
    database: DatabaseConfig
    redis: RedisConfig
    logger: LoggerConfig
    embed: EmbedConfig
    parser: ParserConfig
    llm: LLMConfig
    nlp: NlpConfig
    kbot: KBotConfig