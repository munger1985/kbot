"""入库和手工重处理共享的规范解析计划。"""

from dataclasses import dataclass
from typing import Any

from knowledge_core.domain.parse_settings import normalize_collection_parse_policy
from knowledge_core.parsing import canonical_json_hash


FORBIDDEN_PARSE_POLICY_KEYS = frozenset(
    {
        "embedding",
        "embedding_model",
        "embedding_model_id",
        "embedding_served_model_name",
        "txt_embed_model",
        "txt_embedding_model",
        "query_vector",
        "models",
    }
)


@dataclass(frozen=True)
class ParsePlan:
    policy: dict[str, Any]
    fingerprint: str
    view_kind: str
    parser_name: str


def validate_parse_policy_overrides(overrides: dict[str, Any]) -> None:
    forbidden_keys = FORBIDDEN_PARSE_POLICY_KEYS.intersection(overrides)
    if forbidden_keys:
        raise ValueError(
            "解析策略不能选择或生成检索向量："
            + ", ".join(sorted(forbidden_keys))
        )


def build_parse_plan(*, collection, version, overrides: dict[str, Any]) -> ParsePlan:
    policy: dict[str, Any] = {
        "pipeline": "kc-docling-structure/v1",
        "atom_ir_schema": "kc-atom/v1",
        "structure_ir_schema": "kc-structure/v1",
        "evidence_manifest_schema": "kc-evidence-manifest/v1",
        "quality_gate": "kc-structure-quality/v1",
    }
    policy.update(overrides)
    policy.update(
        normalize_collection_parse_policy(
            dict(getattr(collection, "parse_policy_json", None) or {})
        )
    )
    collection_models = dict(collection.models_json or {})
    policy["models"] = {
        "parser_vlm": collection_models["parser_vlm"]
    } if collection_models.get("parser_vlm") else {}
    if overrides.get("ocr_model"):
        policy["ocr_model"] = overrides["ocr_model"]
    if policy.get("ocr_model"):
        policy["do_ocr"] = False
        policy["ocr_provider"] = "DEEPSEEK_OCR"
    strategy = str(policy.get("parse_strategy", "AUTO")).upper()
    supports_page_visual = version.detected_mime_type in {
        "application/pdf",
        "image/png",
        "image/jpeg",
        "image/tiff",
        "application/vnd.ms-powerpoint",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    }
    if (
        supports_page_visual
        and policy.get("models", {}).get("parser_vlm")
        and strategy != "TEXT"
    ):
        view_kind = "VISUAL" if strategy == "VISUAL" else "HYBRID"
        parser_name = "kc-adaptive-visual-pipeline"
    elif policy.get("ocr_model"):
        view_kind = "TEXT"
        parser_name = "kc-deepseek-ocr-pipeline"
    else:
        view_kind = "TEXT"
        parser_name = "kc-docling-pipeline"
    return ParsePlan(
        policy=policy,
        fingerprint=canonical_json_hash(policy),
        view_kind=view_kind,
        parser_name=parser_name,
    )
