"""Collection 级解析设置。"""

from typing import Any


DEFAULT_COLLECTION_PARSE_POLICY: dict[str, Any] = {
    "parse_strategy": "AUTO",
    "ocr_engine": "tesseract",
    "image_scale": 2.0,
    "detect_table_structure": True,
}

_DERIVED_OR_RETIRED_KEYS = {
    "do_ocr",
    "force_full_page_ocr",
    "extract_page_images",
    "extract_picture_images",
    "visual_min_text_characters",
    "visual_min_mean_confidence",
    "visual_max_gibberish_ratio",
}


def normalize_collection_parse_policy(
    policy: dict[str, Any],
) -> dict[str, Any]:
    unknown = (
        set(policy)
        - set(DEFAULT_COLLECTION_PARSE_POLICY)
        - _DERIVED_OR_RETIRED_KEYS
    )
    if unknown:
        raise ValueError(
            f"不支持的知识库解析参数：{', '.join(sorted(unknown))}"
        )
    normalized = {
        **DEFAULT_COLLECTION_PARSE_POLICY,
        **{
            key: value
            for key, value in policy.items()
            if key in DEFAULT_COLLECTION_PARSE_POLICY
        },
    }
    strategy = str(normalized["parse_strategy"]).strip().upper()
    if strategy not in {"TEXT", "AUTO", "VISUAL"}:
        raise ValueError("parse_strategy 必须是 TEXT、AUTO 或 VISUAL")
    normalized["parse_strategy"] = strategy
    engine = str(normalized["ocr_engine"]).strip().lower()
    if engine not in {"tesseract", "easyocr"}:
        raise ValueError("ocr_engine 必须是 tesseract 或 easyocr")
    normalized["ocr_engine"] = engine
    for key in ("detect_table_structure",):
        if not isinstance(normalized[key], bool):
            raise ValueError(f"{key} 必须是布尔值")
    image_scale = float(normalized["image_scale"])
    if not 1.0 <= image_scale <= 4.0:
        raise ValueError("image_scale 必须在 1.0 到 4.0 之间")
    normalized["image_scale"] = image_scale
    normalized.update({
        "do_ocr": strategy != "TEXT",
        "force_full_page_ocr": False,
        "extract_page_images": False,
        "extract_picture_images": strategy != "TEXT",
    })
    return normalized


__all__ = ["DEFAULT_COLLECTION_PARSE_POLICY", "normalize_collection_parse_policy"]
