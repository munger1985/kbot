"""KBot 4.0 共享平台配置与分层 TOML 加载器。"""

from __future__ import annotations

import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeVar

import tomli
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


class PlatformConfig(BaseModel):
    """所有服务共享的平台标识。"""

    app_id: int = Field(default=1, ge=1)
    version: str = "4.0.0"
    debug: bool = False


class ServiceConfig(BaseModel):
    """独立进程的通用监听配置。"""

    service_name: str
    service_version: str = "4.0.0"
    service_host: str = "0.0.0.0"
    service_port: int = Field(ge=1, le=65535)

    @property
    def service_url(self) -> str:
        host = (
            "127.0.0.1"
            if self.service_host in {"0.0.0.0", "::"}
            else self.service_host
        )
        return f"http://{host}:{self.service_port}"


class ServiceDependencyConfig(BaseModel):
    """跨服务 HTTP 依赖。"""

    base_url: str
    audience: str
    timeout_seconds: int = Field(default=120, ge=1, le=3600)


class LogConfig(BaseModel):
    """所有进程共享的日志策略。"""

    level: str = "INFO"
    dir: str = "./logs"
    rotation: str = "100 MB"
    retention: str = "10 days"
    api_log_enabled: bool = True


class PortalApiKeyConfig(BaseModel):
    """API Key 的非敏感注册信息；明文 Key 不得进入配置文件。"""

    key_id: str = Field(min_length=3, max_length=64)
    client_id: str = Field(min_length=1, max_length=128)
    key_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    enabled: bool = True
    expires_at: datetime | None = None


class SecurityConfig(BaseModel):
    """公开 API 与内部服务认证配置。"""

    api_key_pepper_env: str = "KBOT_API_KEY_PEPPER"
    internal_service_token_env: str = "KBOT_INTERNAL_SERVICE_TOKEN"
    internal_jwt_secret_env: str = "KBOT_INTERNAL_JWT_SECRET"
    internal_jwt_issuer: str = "kbot-platform"
    internal_jwt_ttl_seconds: int = Field(default=60, ge=15, le=300)
    internal_jwt_clock_skew_seconds: int = Field(default=5, ge=0, le=30)
    portal_api_keys: list[PortalApiKeyConfig] = Field(default_factory=list)
    model_api_keys: list[PortalApiKeyConfig] = Field(default_factory=list)


class OracleConfig(BaseModel):
    """Oracle 连接配置；密码只允许通过指定环境变量注入。"""

    username: str = "kbot"
    password_env: str = "KBOT_ORACLE_PASSWORD"
    host: str = "localhost"
    port: int = Field(default=1521, ge=1, le=65535)
    service_name: str = "kbot4"

    def require_password(self) -> str:
        value = os.getenv(self.password_env)
        if not value:
            raise RuntimeError(f"数据库密码环境变量 {self.password_env} 未设置")
        return value


class SQLAlchemyConfig(BaseModel):
    """SQLAlchemy 连接池配置。"""

    echo: bool = False
    pool_size: int = Field(default=10, ge=1, le=100)
    pool_timeout: int = Field(default=60, ge=1, le=300)
    max_overflow: int = Field(default=20, ge=0, le=100)
    pool_pre_ping: bool = True
    pool_use_lifo: bool = True
    pool_recycle: int = Field(default=1800, ge=60, le=86400)


class DatabaseConfig(BaseModel):
    oracle: OracleConfig = Field(default_factory=OracleConfig)
    sqlalchemy: SQLAlchemyConfig = Field(default_factory=SQLAlchemyConfig)


class VectorConfig(BaseModel):
    """跨 KC 与模型服务锁定的唯一向量维度。"""

    dimensions: int = Field(default=1536, ge=1, le=65536)


class Settings(BaseModel):
    """仅包含可被所有独立服务复用的配置。"""

    model_config = ConfigDict(extra="forbid")

    environment: str = "development"
    config_dir: str = "configuration"
    platform: PlatformConfig = Field(default_factory=PlatformConfig)
    log: LogConfig = Field(default_factory=LogConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    vector: VectorConfig = Field(default_factory=VectorConfig)

    def is_development(self) -> bool:
        return self.environment.lower() in {"dev", "development", "debug"}

    def is_production(self) -> bool:
        return self.environment.lower() in {"prod", "production", "live"}

    def is_testing(self) -> bool:
        return self.environment.lower() in {"test", "testing", "staging"}


SettingsT = TypeVar("SettingsT", bound=Settings)


def _load_toml(path: Path, *, required: bool = False) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"配置文件不存在：{path}")
        return {}
    with path.open("rb") as stream:
        return tomli.load(stream)


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = base.copy()
    for key, value in update.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_settings(
    model: type[SettingsT],
    *,
    service: str | None = None,
    config_dir: str | Path | None = None,
    environment: str | None = None,
) -> SettingsT:
    """按共享基座、共享环境、服务基座、服务环境的顺序加载配置。"""

    # 进程环境始终优先；.env 只为本地开发或显式 EnvironmentFile 补值。
    env_file = Path(os.getenv("ENV_FILE", ".env"))
    if env_file.exists():
        load_dotenv(env_file, override=False)

    resolved_dir = Path(
        config_dir or os.getenv("CONFIG_DIR") or "configuration"
    ).resolve()
    resolved_environment = environment or os.getenv(
        "ENVIRONMENT", "development"
    )
    merged = _deep_merge(
        _load_toml(resolved_dir / "base.toml", required=True),
        _load_toml(resolved_dir / f"{resolved_environment}.toml"),
    )
    if service:
        service_dir = resolved_dir / "services" / service
        merged = _deep_merge(
            merged,
            _load_toml(service_dir / "base.toml", required=True),
        )
        merged = _deep_merge(
            merged,
            _load_toml(service_dir / f"{resolved_environment}.toml"),
        )
    merged["environment"] = resolved_environment
    merged["config_dir"] = str(resolved_dir)
    return model.model_validate(merged)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings(Settings)


def get_platform_config() -> PlatformConfig:
    return get_settings().platform


def get_log_config() -> LogConfig:
    return get_settings().log


def get_security_config() -> SecurityConfig:
    return get_settings().security


def get_oracle_config() -> OracleConfig:
    return get_settings().database.oracle


def get_sqlalchemy_config() -> SQLAlchemyConfig:
    return get_settings().database.sqlalchemy
