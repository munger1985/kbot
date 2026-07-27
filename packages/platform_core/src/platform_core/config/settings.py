"""KBot 4.0 共享平台配置与单一 TOML 加载器。"""

from __future__ import annotations

import base64
import hashlib
import hmac
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
    service_identity_jwt_secret_env: str = (
        "KBOT_SERVICE_IDENTITY_JWT_SECRET"
    )
    internal_jwt_issuer: str = "kbot-platform"
    internal_jwt_ttl_seconds: int = Field(default=60, ge=15, le=300)
    service_identity_jwt_ttl_seconds: int = Field(
        default=60, ge=15, le=300
    )
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


def _prepare_runtime_secrets() -> None:
    """从单一主密钥派生进程所需的用途隔离 Secret。"""

    master_key = os.getenv("KBOT_MASTER_KEY")
    if not master_key:
        return
    if len(master_key.encode("utf-8")) < 32:
        raise ValueError("KBOT_MASTER_KEY 必须至少为 32 字节")
    targets = (
        "KBOT_AUTH_ENCRYPTION_KEY",
        "KBOT_API_KEY_PEPPER",
        "KBOT_INTERNAL_SERVICE_TOKEN",
        "KBOT_INTERNAL_JWT_SECRET",
        "KBOT_SERVICE_IDENTITY_JWT_SECRET",
        "KBOT_AIOPS_CURSOR_SECRET",
        "KBOT_AIOPS_DIAGNOSTIC_GRANT_SECRET",
        "KBOT_AIOPS_MUTATION_GRANT_SECRET",
        "KBOT_AIOPS_WEBHOOK_KEY_SECRET",
    )
    for target in targets:
        if os.getenv(target):
            continue
        digest = hmac.new(
            master_key.encode("utf-8"),
            f"kbot4:{target}".encode("utf-8"),
            hashlib.sha256,
        ).digest()
        os.environ[target] = base64.urlsafe_b64encode(digest).decode("ascii")


def _process_settings(
    topology: dict[str, Any], service: str
) -> dict[str, Any]:
    """把拓扑中的进程身份注入所属服务配置段。"""

    result: dict[str, Any] = {}
    for process in topology.get("processes") or ():
        if process.get("service_config") != service:
            continue
        section = str(process["config_section"])
        values = {"service_name": process["service_name"]}
        if "port" in process:
            values["service_port"] = process["port"]
        result[section] = values
    return result


def _endpoint_catalog(
    topology: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """从进程拓扑生成稳定的服务发现目录。"""

    processes = {
        item["process_key"]: item
        for item in topology.get("processes") or ()
    }
    catalog: dict[str, dict[str, Any]] = {}
    for endpoint, process_key in (topology.get("endpoints") or {}).items():
        process = processes.get(process_key)
        if not process or "port" not in process:
            raise ValueError(
                f"拓扑端点 {endpoint} 引用了无 HTTP 端口的进程 {process_key}"
            )
        catalog[endpoint] = {
            "base_url": f"http://127.0.0.1:{process['port']}",
            "audience": process["service_name"],
        }
    for endpoint, value in overrides.items():
        if endpoint not in catalog:
            raise ValueError(f"部署配置覆盖了未知端点：{endpoint}")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"端点覆盖 {endpoint} 必须是非空 URL")
        catalog[endpoint]["base_url"] = value.rstrip("/")
    return catalog


def _resolve_dependencies(
    bindings: dict[str, Any],
    endpoints: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """把依赖绑定中的端点别名展开为客户端配置。"""

    result: dict[str, Any] = {}
    for key, value in bindings.items():
        if not isinstance(value, dict):
            raise ValueError(f"依赖绑定 {key} 必须是对象")
        endpoint = value.get("endpoint")
        if endpoint:
            if endpoint not in endpoints:
                raise ValueError(f"依赖绑定 {key} 引用了未知端点：{endpoint}")
            result[key] = _deep_merge(
                endpoints[endpoint],
                {
                    name: item
                    for name, item in value.items()
                    if name != "endpoint"
                },
            )
        else:
            result[key] = _resolve_dependencies(value, endpoints)
    return result


def load_settings(
    model: type[SettingsT],
    *,
    service: str | None = None,
    config_file: str | Path | None = None,
    environment: str | None = None,
) -> SettingsT:
    """加载单一部署文件，并把少量部署参数映射到服务配置模型。"""

    # 进程环境始终优先；.env 只为本地开发或显式 EnvironmentFile 补值。
    env_file = Path(os.getenv("ENV_FILE", ".env"))
    if env_file.exists():
        load_dotenv(env_file, override=False)
    _prepare_runtime_secrets()

    resolved_file = Path(
        config_file
        or os.getenv("KBOT_CONFIG_FILE")
        or "configuration/kbot.toml"
    ).resolve()
    deployment = _load_toml(resolved_file, required=True)
    unknown_keys = set(deployment) - {
        "environment",
        "app_id",
        "data_dir",
        "log_dir",
        "embedding_dimension",
        "development_auth_bypass",
        "portal_api_keys",
        "model_api_keys",
        "database",
        "endpoints",
        "paths",
        "integrations",
    }
    if unknown_keys:
        names = "、".join(sorted(unknown_keys))
        raise ValueError(f"部署配置包含未知顶层字段：{names}")

    resolved_environment = (
        environment
        or os.getenv("ENVIRONMENT")
        or deployment.get("environment")
        or "development"
    )
    if resolved_environment.lower() in {"prod", "production", "live"}:
        required_fields = {
            "data_dir",
            "log_dir",
            "embedding_dimension",
            "database",
            "portal_api_keys",
        }
        missing = sorted(required_fields - set(deployment))
        if missing:
            raise ValueError(f"生产配置缺少必填字段：{missing}")
        required_database = {"host", "username", "service_name"}
        database_value = deployment.get("database")
        missing_database = sorted(
            required_database
            - set(database_value if isinstance(database_value, dict) else {})
        )
        if missing_database:
            raise ValueError(
                f"生产数据库配置缺少字段：{missing_database}"
            )
        if deployment.get("development_auth_bypass"):
            raise ValueError("生产环境禁止开启开发认证绕过")
        if not deployment.get("portal_api_keys"):
            raise ValueError("生产环境至少需要一个 Portal API Key 摘要")
    data_dir = Path(str(deployment.get("data_dir") or "./var/data"))
    log_dir = str(deployment.get("log_dir") or "./var/log")
    database = deployment.get("database") or {}
    endpoints_override = deployment.get("endpoints") or {}
    paths = deployment.get("paths") or {}
    integrations = deployment.get("integrations") or {}
    for name, value in (
        ("database", database),
        ("endpoints", endpoints_override),
        ("paths", paths),
        ("integrations", integrations),
    ):
        if not isinstance(value, dict):
            raise ValueError(f"kbot.toml 的 [{name}] 必须是对象")

    resource_dir = Path(
        os.getenv("KBOT_RESOURCE_DIR")
        or Path.cwd() / "resources"
    ).resolve()
    topology = _load_toml(resource_dir / "topology.toml", required=True)
    merged: dict[str, Any] = {
        "platform": {
            "app_id": deployment.get("app_id", 1),
            "version": "4.0.0",
            "debug": resolved_environment.lower()
            in {"dev", "development", "debug"},
        },
        "log": {
            "level": (
                "DEBUG"
                if resolved_environment.lower()
                in {"dev", "development", "debug"}
                else "INFO"
            ),
            "dir": log_dir,
        },
        "security": {
            "portal_api_keys": deployment.get("portal_api_keys") or [],
            "model_api_keys": deployment.get("model_api_keys") or [],
        },
        "database": {"oracle": database},
        "vector": {
            "dimensions": deployment.get("embedding_dimension", 1536)
        },
    }
    if service:
        merged = _deep_merge(merged, _process_settings(topology, service))
        endpoints = _endpoint_catalog(topology, endpoints_override)
        bindings = (topology.get("dependencies") or {}).get(service) or {}
        merged = _deep_merge(
            merged, _resolve_dependencies(bindings, endpoints)
        )
        service_paths: dict[str, Any] = {
            "knowledge_core": {
                "storage": {
                    "local_object_storage_path": str(
                        data_dir / "knowledge_core"
                    )
                },
                "parser": {
                    "local_artifacts_path": str(
                        paths.get("docling_models")
                        or data_dir / "models" / "docling_models"
                    )
                },
            },
            "agent_runtime": {
                "attachments": {
                    "local_storage_path": str(data_dir / "agent_runtime")
                }
            },
            "model_serving": {
                "embedding": {"cache_dir": str(data_dir / "models")}
            },
            "aiops_agent": {
                "monitoring": {
                    "payload_store_root": str(
                        data_dir / "aiops" / "monitor_payloads"
                    )
                }
            },
            "main_api": {
                "api": {
                    "allowed_origins": (
                        [
                            "http://127.0.0.1:8080",
                            "http://localhost:8080",
                        ]
                        if resolved_environment.lower()
                        in {"dev", "development", "debug"}
                        else []
                    ),
                    "test_auth_bypass_enabled": bool(
                        deployment.get("development_auth_bypass", False)
                    ),
                }
            },
        }
        merged = _deep_merge(merged, service_paths.get(service, {}))
        if service == "agent_runtime" and integrations.get("mcp_data"):
            merged = _deep_merge(
                merged, {"ask_data_api": integrations["mcp_data"]}
            )
        if service == "knowledge_core" and integrations.get(
            "deepseek_ocr"
        ):
            deepseek = dict(integrations["deepseek_ocr"])
            ocr_model = deepseek.pop("ocr_model", None)
            merged = _deep_merge(merged, {"dsocr": deepseek})
            if ocr_model:
                merged = _deep_merge(
                    merged, {"parse_policy": {"ocr_model": ocr_model}}
                )

    merged["environment"] = resolved_environment
    merged["config_dir"] = str(resolved_file.parent)
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
