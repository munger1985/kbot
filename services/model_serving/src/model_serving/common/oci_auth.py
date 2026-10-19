"""OCI API Key 认证配置的解析与启动前校验。"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlsplit

import httpx
import requests


def oci_sdk_service_endpoint(raw: Any) -> str:
    """校验并返回 OCI SDK 所需的服务 Origin。"""
    endpoint = str(raw or "").strip().rstrip("/")
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("OCI API Endpoint 必须是无路径、查询参数和片段的 HTTP(S) 服务 Origin")
    return f"{parsed.scheme.lower()}://{parsed.netloc}"


def _oci_signer(config: dict[str, Any]):
    import oci

    return oci.signer.Signer(
        tenancy=config["tenancy"],
        user=config["user"],
        fingerprint=config["fingerprint"],
        private_key_file_location=config.get("key_file"),
        pass_phrase=config.get("pass_phrase"),
        private_key_content=config.get("key_content"),
    )


class ModelConfigOciAuth(httpx.Auth):
    """把 OCI SDK signer 接入 OpenAI SDK 使用的 HTTPX 请求。"""

    def __init__(self, config: dict[str, Any]) -> None:
        self._signer = _oci_signer(dict(config))

    def auth_flow(self, request: httpx.Request):
        """对 OpenAI SDK 已序列化的请求正文签署 OCI IAM 请求头。"""
        try:
            content = request.content
        except httpx.RequestNotRead:
            content = request.read()
        request.headers.pop("Authorization", None)
        prepared = requests.Request(
            method=request.method,
            url=str(request.url),
            headers=dict(request.headers),
            data=content,
        ).prepare()
        self._signer.do_request_sign(prepared)
        request.headers.update(prepared.headers)
        yield request


def validated_oci_config(raw: Any) -> dict[str, Any]:
    """返回已校验的 OCI 配置，错误信息只包含字段名。"""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("OCI config_file 不是有效的 JSON Object") from exc
    if not isinstance(raw, dict):
        raise ValueError("OCI config_file 必须是 JSON Object")

    config = dict(raw)
    try:
        import oci

        oci.config.validate_config(config)
        _oci_signer(config)
    except oci.exceptions.InvalidConfig as exc:
        details = exc.args[0] if exc.args else {}
        fields = sorted(details) if isinstance(details, dict) else []
        suffix = f"：{', '.join(fields)}" if fields else ""
        raise ValueError(f"OCI config_file 字段缺失或格式错误{suffix}") from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("OCI config_file 的 API Key 认证材料无效") from exc
    return config


def public_model_params(raw: Any) -> dict[str, Any]:
    """删除模型目录中不允许对外投影的嵌套认证材料。"""
    if not isinstance(raw, dict):
        return {}
    return {
        key: value
        for key, value in raw.items()
        if key != "config_file"
    }


__all__ = [
    "ModelConfigOciAuth",
    "oci_sdk_service_endpoint",
    "public_model_params",
    "validated_oci_config",
]
