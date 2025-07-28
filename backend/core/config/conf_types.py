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

class LoggerConfig(TypedDict):
    level: str
    dir: str
    rotation: str
    retention: str

class EmbedConfig(TypedDict):
    max_workers: int
    check_interval: int
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

class RankerConfig(TypedDict):
    max_tokens: int

class NlpConfig(TypedDict):
    model_name: str

class KBotConfig(TypedDict):
    file_root_path: str
    parallel_workers: int

class AppSettings(TypedDict):
    app: AppConfig
    database: DatabaseConfig
    logger: LoggerConfig
    embed: EmbedConfig
    llm: LLMConfig
    reranker: RankerConfig
    nlp: NlpConfig
    kbot: KBotConfig