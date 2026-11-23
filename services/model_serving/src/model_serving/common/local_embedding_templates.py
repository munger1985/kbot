"""本地 Embedding 输入规则与表示空间标识共用同一来源。"""

from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json


def qwen_embedding_text(text: str, *, is_query: bool, instruction: str | None) -> str:
    return f"Instruct: {instruction}\nQuery: {text}" if is_query else text


def bge_embedding_text(text: str, *, is_query: bool, instruction: str | None) -> str:
    return f"{instruction}{text}" if is_query and instruction else text


def _template_id(contract: str, rules: dict) -> str:
    wire = json.dumps(rules, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{contract}:sha256:{sha256(wire.encode()).hexdigest()}"


def text_input_template(config) -> str:
    if config.provider == "local_qwen":
        formatter, instruction = qwen_embedding_text, config.instruction
    elif config.provider == "local_bge":
        formatter, instruction = bge_embedding_text, config.query_instruction
    else:
        raise ValueError("此文本模型不支持自动生成输入模板")
    return _template_id("local-text-embedding/v1", {
        "provider": config.provider,
        "document": formatter("{text}", is_query=False, instruction=instruction),
        "query": formatter("{text}", is_query=True, instruction=instruction),
        "truncation": True, "max_tokens": config.max_tokens,
    })


def colqwen_input_template() -> str:
    try:
        processor_version = version("colpali-engine")
    except PackageNotFoundError as exc:
        raise ValueError("ColQwen2 未安装预处理运行库") from exc
    return _template_id("local-colqwen2/v1", {
        "text": "ColQwen2Processor.process_queries", "image": "RGB -> ColQwen2Processor.process_images",
        "colpali_engine": processor_version,
    })
