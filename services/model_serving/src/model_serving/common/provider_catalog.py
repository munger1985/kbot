"""由代码控制的模型 Provider 目录与参数校验。"""

from dataclasses import dataclass
from typing import Any

from platform_core.contracts import ModelProviderOption
from platform_core.dictionary import ModelCategory

from model_serving.common.oci_auth import validated_oci_config


@dataclass(frozen=True, slots=True)
class _ProviderSchema:
    required_fields: tuple[str, ...]
    allowed_model_params: tuple[str, ...]
    supports_tool_calling: bool = False
    max_context_tokens: int | None = None
    secret_fields: tuple[str, ...] = ("api_key",)


_COMMON_REMOTE = ("timeout", "max_retries")
_LLM_PARAMS = (
    "temperature", "max_tokens", "top_p", "frequency_penalty",
    "presence_penalty", "timeout", "compartment_id", "config_file",
)
_EMBEDDING_PARAMS = (
    "embedding_dimension", "model_path", "device", "max_tokens", "timeout",
    "compartment_id", "config_file", "batch_size", "normalize",
)
_VLM_PARAMS = ("max_tokens", "temperature", *_COMMON_REMOTE)
_VISUAL_PARAMS = ("model_path", "device", "dimension", *_COMMON_REMOTE)

PROVIDER_SCHEMAS: dict[tuple[int, str], _ProviderSchema] = {
    **{
        (ModelCategory.LLM.value, provider): _ProviderSchema(
            ("api_endpoint", "api_key"), _LLM_PARAMS,
            supports_tool_calling=True, max_context_tokens=131072,
        )
        for provider in ("api_deepseek", "api_qwen", "chatgpt")
    },
    (ModelCategory.LLM.value, "oci"): _ProviderSchema(
        ("api_endpoint", "model_params.compartment_id", "model_params.config_file"),
        _LLM_PARAMS, supports_tool_calling=True, max_context_tokens=131072,
        secret_fields=("model_params.config_file",),
    ),
    (ModelCategory.TXT_EMBEDDING.value, "local_bge"): _ProviderSchema(
        ("model_params.model_path", "model_params.embedding_dimension"),
        _EMBEDDING_PARAMS,
    ),
    (ModelCategory.TXT_EMBEDDING.value, "local_qwen"): _ProviderSchema(
        ("model_params.model_path", "model_params.embedding_dimension"),
        _EMBEDDING_PARAMS,
    ),
    **{
        (ModelCategory.TXT_EMBEDDING.value, provider): _ProviderSchema(
            ("api_endpoint", "api_key", "model_params.embedding_dimension"),
            _EMBEDDING_PARAMS,
        )
        for provider in ("api_qwen", "chatgpt")
    },
    (ModelCategory.TXT_EMBEDDING.value, "oci"): _ProviderSchema(
        (
            "api_endpoint", "model_params.compartment_id",
            "model_params.config_file", "model_params.embedding_dimension",
        ),
        _EMBEDDING_PARAMS,
        secret_fields=("model_params.config_file",),
    ),
    (ModelCategory.IMG_EMBEDDING.value, "local_qwen"): _ProviderSchema(
        ("model_params.model_path",), _VISUAL_PARAMS,
    ),
    **{
        (ModelCategory.VLM.value, provider): _ProviderSchema(
            ("api_endpoint", "api_key"), _VLM_PARAMS,
            max_context_tokens=32768,
        )
        for provider in ("api_qwen", "chatgpt")
    },
}


def list_provider_options(*, category: int) -> list[ModelProviderOption]:
    """列出一个模型类别可以使用的 Provider，不返回任何凭据值。"""
    return [
        ModelProviderOption(
            category=provider_category,
            provider=provider,
            required_fields=schema.required_fields,
            secret_fields=schema.secret_fields,
            allowed_model_params=schema.allowed_model_params,
            supports_tool_calling=schema.supports_tool_calling,
            max_context_tokens=schema.max_context_tokens,
        )
        for (provider_category, provider), schema in sorted(PROVIDER_SCHEMAS.items())
        if provider_category == category
    ]


def validate_provider_config(values: dict[str, Any]) -> None:
    """按类别和 Provider 拒绝未知参数并校验必要字段。"""
    category = int(values.get("category") or 0)
    provider = str(values.get("provider") or "").strip().lower()
    schema = PROVIDER_SCHEMAS.get((category, provider))
    if schema is None:
        raise ValueError(f"模型类别 {category} 不支持 Provider：{provider}")
    params = values.get("model_params") or {}
    if not isinstance(params, dict):
        raise ValueError("model_params 必须是对象")
    unknown = sorted(set(params) - set(schema.allowed_model_params))
    if unknown:
        raise ValueError(f"model_params 包含未知参数：{unknown}")
    missing: list[str] = []
    for path in schema.required_fields:
        if path.startswith("model_params."):
            value = params.get(path.removeprefix("model_params."))
        else:
            value = values.get(path)
        if value is None or value == "":
            missing.append(path)
    if missing:
        raise ValueError(f"Provider 配置缺少必要字段：{missing}")
    if provider == "oci":
        validated_oci_config(params.get("config_file"))
