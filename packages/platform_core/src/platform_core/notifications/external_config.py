"""独立外部投递进程的有界配置。"""

from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExternalDeliveryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # 拓扑注入的进程标识；Worker 不监听 HTTP 端口。
    service_name: str = "external-notification-worker"
    service_host: str = "127.0.0.1"
    enabled: bool = False
    public_origin: str = ""
    token_url: str = "https://accounts.feishu.cn/oauth/v3/token"
    message_origin: str = "https://open.feishu.cn"
    # 仅供注入假凭据的自动化环境；生产调用方必须拒绝开启。
    fixture_mode: bool = False
    fixture_allowed_hosts: list[str] = Field(default_factory=lambda: ["127.0.0.1", "localhost"])
    poll_seconds: float = Field(default=1, ge=0.1, le=60)
    request_timeout_seconds: float = Field(default=10, ge=1, le=30)
    lease_seconds: int = Field(default=60, ge=40, le=600)
    max_attempts: int = Field(default=5, ge=1, le=20)
    retry_base_seconds: float = Field(default=5, ge=1, le=120)
    retry_max_seconds: float = Field(default=900, ge=60, le=3600)
    dedupe_safe_seconds: int = Field(default=3300, ge=60, le=3300)
    connection_interval_seconds: float = Field(default=0.2, ge=0.1, le=60)
    target_interval_seconds: float = Field(default=1, ge=1, le=3600)
    global_interval_seconds: float = Field(default=0.1, ge=0.01, le=60)
    job_ttl_seconds: int = Field(default=3600, ge=60, le=86400)
    entry_ttl_days: int = Field(default=30, ge=1, le=90)
    # 仅对 INFO／WARNING 的新告警、恢复生效；UTC 时段支持跨日，默认不静默、不合并。
    quiet_start_utc_hour: int | None = Field(default=None, ge=0, le=23)
    quiet_end_utc_hour: int | None = Field(default=None, ge=0, le=23)
    coalesce_seconds: int = Field(default=0, ge=0, le=300)
    verification_ttl_seconds: int = Field(default=600, ge=60, le=1800)
    verification_max_attempts: int = Field(default=5, ge=1, le=10)
    verification_cooldown_seconds: int = Field(default=60, ge=30, le=3600)

    @model_validator(mode="after")
    def endpoints(self):
        if ((self.quiet_start_utc_hour is None) != (self.quiet_end_utc_hour is None)
            or (self.quiet_start_utc_hour is not None and self.quiet_start_utc_hour == self.quiet_end_utc_hour)):
            raise ValueError("NOTIFICATION_QUIET_WINDOW_INVALID")
        token, message = urlsplit(self.token_url), urlsplit(self.message_origin)
        if not self.fixture_mode:
            if self.token_url != "https://accounts.feishu.cn/oauth/v3/token" or self.message_origin != "https://open.feishu.cn":
                raise ValueError("NOTIFICATION_PROVIDER_ENDPOINT_DENIED")
        elif any(url.hostname not in self.fixture_allowed_hosts or url.scheme not in {"http", "https"}
                 or url.username or url.password for url in (token, message)):
            raise ValueError("NOTIFICATION_FIXTURE_ENDPOINT_DENIED")
        if message.path or message.query or message.fragment:
            raise ValueError("NOTIFICATION_MESSAGE_ORIGIN_INVALID")
        if self.public_origin:
            origin = urlsplit(self.public_origin)
            if (origin.scheme != "https" and not (self.fixture_mode and origin.scheme == "http")) or not origin.netloc or origin.path or origin.query or origin.fragment or origin.username:
                raise ValueError("NOTIFICATION_PUBLIC_ORIGIN_INVALID")
        if self.enabled and not self.public_origin:
            raise ValueError("NOTIFICATION_PUBLIC_ORIGIN_REQUIRED")
        if self.lease_seconds <= 2 * self.request_timeout_seconds + 5:
            raise ValueError("NOTIFICATION_LEASE_TOO_SHORT")
        return self

    def validate_runtime(self, *, production: bool) -> None:
        if production and self.fixture_mode:
            raise ValueError("NOTIFICATION_PRODUCTION_FIXTURE_DENIED")
