"""校验 KM Asset 统一搜索的部署边界与托管模型。"""

from pathlib import Path

from data_query.application.managed_datasets import km_asset_definition
from data_query.contracts import SemanticModelDefinition


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    errors: list[str] = []
    model = SemanticModelDefinition.model_validate(
        km_asset_definition(schema_name="KBOTUI_DEV")
    )
    dataset = model.datasets[0]
    dimensions = {item.name for item in model.dimensions}
    if dataset.physical_object != "KBOT_V_KM_ASSET_SEARCHABLE":
        errors.append("托管模型未绑定可搜索视图")
    if dataset.scope_column != "DOMAIN_ID":
        errors.append("托管模型缺少 Domain 强制范围")
    if "topic" in dimensions:
        errors.append("语义主题仍被错误暴露为精确问数维度")
    if "asset_status" in dimensions:
        errors.append("READY 系统边界仍被错误暴露为用户筛选字段")

    ddl = (ROOT / "database/oracle/km_asset_app/001_km_asset.sql").read_text(
        encoding="utf-8"
    )
    required = (
        "CREATE OR REPLACE VIEW KBOT_V_KM_ASSET_SEARCHABLE",
        "A.INGESTION_STATUS = 'READY'",
        "R.STATUS = 'READY'",
        "R.KC_BUNDLE_REVISION_ID = A.KC_BUNDLE_REVISION_ID",
        "A.KC_BUNDLE_ID IS NOT NULL",
    )
    for text in required:
        if text not in ddl:
            errors.append(f"可搜索视图缺少约束：{text}")

    prompts = (
        ROOT / "packages/platform_core/src/platform_core/resources/prompts.toml"
    ).read_text(encoding="utf-8")
    if 'prompt_key = "agent_runtime.km_asset_search_plan"' not in prompts:
        errors.append("缺少 AssetSearchPlan.v1 Prompt")
    if "不得把 ingestion_status 加入 criteria" not in prompts:
        errors.append("Prompt 未冻结 READY 系统边界")

    if errors:
        print("KM Asset 统一搜索部署检查失败：")
        print("\n".join(errors))
        return 1
    print("KM Asset 统一搜索部署检查通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
