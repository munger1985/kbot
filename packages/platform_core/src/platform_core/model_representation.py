"""模型目录与推理绑定共用的向量表示规格。"""

from typing import Literal

from .dictionary import EmbeddingProvider, ModelCategory


ImageSearchDirection = Literal["TEXT_TO_IMAGE", "IMAGE_TO_IMAGE"]
_IMAGE_SEARCH_DIRECTIONS = {"TEXT_TO_IMAGE", "IMAGE_TO_IMAGE"}


def _representation_configuration(data, kind):
    """校验目录声明的维度与检索方向。"""
    params = data.get("model_params") or {}
    visual = kind == "MULTIVECTOR"
    if visual and data.get("provider") != EmbeddingProvider.LOCAL_QWEN.value:
        raise ValueError("此视觉模型不提供原生多向量")
    if not isinstance(params, dict):
        raise ValueError("模型参数不是对象")
    dimension = params.get("dimension" if visual else "embedding_dimension")
    directions = params.get("retrieval_directions") if visual else ["TEXT_TO_TEXT"]
    permitted = _IMAGE_SEARCH_DIRECTIONS | {"IMAGE_TO_TEXT", "TEXT_TO_TEXT"}
    if (type(dimension) is not int or not 0 < dimension <= 65536
            or not isinstance(directions, (tuple, list)) or not directions
            or any(not isinstance(item, str) or item not in permitted for item in directions)
            or (visual and "IMAGE_TO_IMAGE" not in directions)):
        raise ValueError("模型未登记完整表示规格")
    return {"dimension": dimension, "directions": sorted(set(directions)),
            "kind": kind, "input_kind": "IMAGE" if visual else "TEXT",
            "similarity": "DOT" if visual else "COSINE", "normalization": "L2"}


def representation_metadata(data, kind):
    """绑定推理需要输入模板与两个稳定版本；本地模型由模型服务先计算。"""
    config = _representation_configuration(data, kind)
    metadata = {key: data["model_params"].get(key)
                for key in ("input_template", "model_revision", "processor_revision")}
    if not all(isinstance(value, str) and value.strip() for value in metadata.values()):
        raise ValueError("模型未登记完整表示规格")
    return {**config, **metadata}


def registered_image_search_directions(data) -> list[ImageSearchDirection] | None:
    """只公开用于检索图片的登记方向；缺少有效规格时不声称支持。"""
    if data.get("category") != ModelCategory.IMG_EMBEDDING:
        return None
    try:
        # 目录只展示声明方向；模板和版本由模型服务在绑定时验证。
        spec = _representation_configuration(data, "MULTIVECTOR")
    except ValueError:
        return None
    return [direction for direction in spec["directions"] if direction in _IMAGE_SEARCH_DIRECTIONS]
