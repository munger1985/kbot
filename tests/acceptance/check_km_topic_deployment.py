"""检查 KM 主题问数代码是否从当前 Python 环境正确加载。"""

from __future__ import annotations

import inspect

import platform_core
from data_query.application.managed_datasets import km_asset_definition
from platform_core.prompts import load_prompt_catalog
from platform_core.prompts.catalog import DEFAULT_PROMPT_CATALOG


EXPECTED_PROMPTS = {
    "agent_runtime.km_asset_intent_route": "1.2.0",
    "agent_runtime.km_asset_context_route": "1.2.0",
    "agent_runtime.data_query_plan": "1.2.0",
    "agent_runtime.km_asset_enumeration_compose": "1.0.0",
}


def main() -> int:
    """打印实际加载位置，并校验 Prompt 与托管模型代码。"""
    catalog = load_prompt_catalog()
    active_versions = {
        item.prompt_key: item.version
        for item in catalog.entries
        if item.active and item.prompt_key in EXPECTED_PROMPTS
    }
    definition = km_asset_definition(schema_name="CHECK_ONLY")
    dimension_names = {
        str(item.get("name"))
        for item in definition.get("dimensions", ())
        if isinstance(item, dict)
    }
    has_topic = "topic" in dimension_names
    has_enumeration_scope = {
        "asset_id", "title", "bundle_id", "bundle_revision_id"
    }.issubset(dimension_names)

    print(f"platform_core = {platform_core.__file__}")
    print(f"prompt_catalog = {DEFAULT_PROMPT_CATALOG}")
    print(
        "managed_datasets = "
        f"{inspect.getsourcefile(km_asset_definition)}"
    )
    for prompt_key, expected_version in EXPECTED_PROMPTS.items():
        actual_version = active_versions.get(prompt_key, "MISSING")
        print(
            f"{prompt_key} = {actual_version} "
            f"(expected {expected_version})"
        )
    print(f"HAS_TOPIC_IN_CODE = {has_topic}")
    print(f"HAS_ENUMERATION_SCOPE_IN_CODE = {has_enumeration_scope}")

    prompts_ok = all(
        active_versions.get(prompt_key) == expected_version
        for prompt_key, expected_version in EXPECTED_PROMPTS.items()
    )
    if prompts_ok and has_topic and has_enumeration_scope:
        print("KM 主题问数代码加载检查通过")
        return 0
    print("KM 主题问数代码加载检查失败：当前环境仍在加载旧代码")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
