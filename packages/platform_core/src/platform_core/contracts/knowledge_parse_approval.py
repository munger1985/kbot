"""审批时解析库绑定的模型；仅使用服务端目录返回的实际版本。"""

import asyncio

from .knowledge_parsing import LibraryParseModels, REQUIRED_PARSE_MODELS, parse_strategy_options


async def available_parse_bindings(models, lookup):
    configured = LibraryParseModels.model_validate(models or {})
    if lookup is None:
        return {}

    async def resolve(role):
        model_id = getattr(configured, role)
        if model_id is None:
            return role, None
        try:
            async with asyncio.timeout(5):
                binding = await lookup(role, model_id)
            if (not isinstance(binding, dict) or str(binding.get("model_id")) != str(model_id)
                    or not isinstance(binding.get("revision"), str) or not binding["revision"]
                    or not isinstance(binding.get("inference_ref"), str) or not binding["inference_ref"]):
                return role, None
            return role, binding
        except (OSError, RuntimeError, ValueError):
            return role, None

    return {role: binding for role, binding in await asyncio.gather(resolve("ocr"), resolve("vlm")) if binding is not None}


def approval_parse_options(bindings, *, executor_ready):
    return [option.model_dump(mode="json") for option in parse_strategy_options(bindings, executor_ready=executor_ready)]


def bindings_for_strategy(strategy, bindings):
    if strategy not in REQUIRED_PARSE_MODELS:
        raise ValueError("请选择有效的解析方案")
    if any(role not in bindings for role in REQUIRED_PARSE_MODELS[strategy]):
        raise ValueError("所选解析方案的库模型未配置或当前不可用，请刷新后重试")
    return {} if strategy == "TEXT" else {role: bindings[role] for role in ("ocr", "vlm") if role in bindings}
