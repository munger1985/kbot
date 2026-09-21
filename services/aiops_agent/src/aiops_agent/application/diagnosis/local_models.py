"""AIOps 生产环境的本地诊断模型约束。"""

from __future__ import annotations

from platform_core.dictionary import (
    LLMProvider,
    is_local_llm_provider,
)


PRODUCTION_ENVIRONMENTS = frozenset({"prod", "production", "live"})
PRODUCTION_AIOPS_LLM_PROVIDER = LLMProvider.LOCAL_DEEPSEEK.value


class LocalModelBindingError(ValueError):
    """生产环境绑定了非本地诊断模型。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def is_production_environment(environment: str | None) -> bool:
    """与平台 Settings.is_production() 使用同一组环境名。"""
    return str(environment or "").strip().lower() in PRODUCTION_ENVIRONMENTS


def require_production_local_llm(
    *,
    environment: str | None,
    provider: str | None,
    role_label: str,
) -> None:
    """生产环境的规划/诊断模型必须是客户近端 DeepSeek。"""
    if not is_production_environment(environment):
        return
    if is_local_llm_provider(provider):
        return
    raise LocalModelBindingError(
        "AIOPS_AGENT_PRODUCTION_LOCAL_MODEL_REQUIRED",
        (
            f"生产环境的{role_label}模型必须绑定本地部署的 DeepSeek"
            f"（provider={PRODUCTION_AIOPS_LLM_PROVIDER}）"
        ),
    )
