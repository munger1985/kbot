from pydantic import BaseModel

class BaseDBConfig(BaseModel):
    host: str
    port: int
    user: str
    password: str
    database: str | None = None

class PGConfig(BaseDBConfig):
    port: int = 5432

class MySQLConfig(BaseDBConfig):
    port: int = 3306
    charset: str = "utf8mb4"

class OracleConfig(BaseModel):
    # Oracle 比较特殊，可能只有 DSN 或者 分体参数
    user: str
    password: str
    host: str | None = None
    port: int = 1521
    service_name: str | None = None
    dsn: str | None = None  # 如果传了 DSN，优先使用 DSN