"""固定案例集合的显式模型配置；不混入资料库或自动借用其他资源。"""

from aiops_agent.application.errors import AIOpsApplicationError
from platform_core.contracts.aiops.case_collection import CASE_MODEL_ROLES


def case_collection_configuration(collection):
    """区分绑定完整与实际模型可用，不把静态配置读取当成运行验收。"""
    models = dict(collection.get("models") or {})
    strategy = str((collection.get("parse_policy") or {}).get("parse_strategy", "AUTO")).upper()
    missing = sorted(role for role in CASE_MODEL_ROLES if not models.get(role))
    return {
        "resource_kind": "DIAGNOSIS_CASE",
        "collection_id": str(collection["collection_id"]),
        "models": models,
        "required_model_roles": sorted(CASE_MODEL_ROLES),
        "missing_model_roles": missing,
        "parse_strategy": strategy,
        "configuration_status": ("TEXT_POLICY_REQUIRED" if strategy not in {"AUTO", "TEXT"}
            else "MODELS_REQUIRED" if missing else "CONFIGURED"),
        "retrieval_requires_publication": True,
    }


def assert_case_text_policy(collection):
    """保留管理员显式策略；案例配置不静默改写 VISUAL 或未知策略。"""
    if case_collection_configuration(collection)["parse_strategy"] not in {"AUTO", "TEXT"}:
        raise AIOpsApplicationError(code="AIOPS_CASE_TEXT_POLICY_REQUIRED",
            message="案例正文需要 AUTO 原生文本或 TEXT 策略，请先在现有 KC 配置中调整",
            status_code=409)
