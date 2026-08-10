"""Collection 级解析设置。"""

from typing import Any


DEFAULT_COLLECTION_PARSE_POLICY: dict[str, Any] = {
    "parse_strategy": "AUTO",
    "do_ocr": True,
    "ocr_engine": "tesseract",
    "image_scale": 2.0,
    "extract_page_images": True,
    "extract_picture_images": True,
    "detect_table_structure": True,
    "visual_min_text_characters": 80,
    "visual_min_mean_confidence": 0.65,
    "visual_max_gibberish_ratio": 0.08,
}


def normalize_collection_parse_policy(
    policy: dict[str, Any],
) -> dict[str, Any]:
    unknown = set(policy) - set(DEFAULT_COLLECTION_PARSE_POLICY)
    if unknown:
        raise ValueError(
            f"不支持的知识库解析参数：{', '.join(sorted(unknown))}"
        )
    normalized = {**DEFAULT_COLLECTION_PARSE_POLICY, **policy}
    strategy = str(normalized["parse_strategy"]).strip().upper()
    if strategy not in {"TEXT", "AUTO", "HYBRID", "VISUAL"}:
        raise ValueError("parse_strategy 必须是 TEXT、AUTO、HYBRID 或 VISUAL")
    normalized["parse_strategy"] = strategy
    engine = str(normalized["ocr_engine"]).strip().lower()
    if engine not in {"tesseract", "easyocr"}:
        raise ValueError("ocr_engine 必须是 tesseract 或 easyocr")
    normalized["ocr_engine"] = engine
    for key in (
        "do_ocr",
        "extract_page_images",
        "extract_picture_images",
        "detect_table_structure",
    ):
        if not isinstance(normalized[key], bool):
            raise ValueError(f"{key} 必须是布尔值")
    image_scale = float(normalized["image_scale"])
    if not 1.0 <= image_scale <= 4.0:
        raise ValueError("image_scale 必须在 1.0 到 4.0 之间")
    normalized["image_scale"] = image_scale
    text_characters = int(normalized["visual_min_text_characters"])
    if not 0 <= text_characters <= 10000:
        raise ValueError("visual_min_text_characters 必须在 0 到 10000 之间")
    normalized["visual_min_text_characters"] = text_characters
    for key in ("visual_min_mean_confidence", "visual_max_gibberish_ratio"):
        value = float(normalized[key])
        if not 0 <= value <= 1:
            raise ValueError(f"{key} 必须在 0 到 1 之间")
        normalized[key] = value
    return normalized


__all__ = ["DEFAULT_COLLECTION_PARSE_POLICY", "normalize_collection_parse_policy"]
