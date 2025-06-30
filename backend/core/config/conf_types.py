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

class LoggingConfig(TypedDict):
    level: str
    path: str
    rotation: str
    retention: str

class ChunkConfig(TypedDict):
    size: int
    overlap: int

class KBotConfig(TypedDict):
    file_root_path: str
    parallel_workers: int

class AppSettings(TypedDict):
    app: AppConfig
    database: DatabaseConfig
    logging: LoggingConfig
    chunk: ChunkConfig
    kbot: KBotConfig