"""OCI API Key 认证配置的解析与启动前校验。"""

from __future__ import annotations

import json
from typing import Any


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
        oci.signer.Signer(
            tenancy=config["tenancy"],
            user=config["user"],
            fingerprint=config["fingerprint"],
            private_key_file_location=config.get("key_file"),
            pass_phrase=config.get("pass_phrase"),
            private_key_content=config.get("key_content"),
        )
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


__all__ = ["public_model_params", "validated_oci_config"]
