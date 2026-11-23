"""依据本地模型制品计算版本；目录里的手填版本不是运行时依据。"""

import asyncio
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
from threading import Lock

from platform_core.dictionary import ModelCategory


_LOCK = Lock()
_PROCESSOR_FILES = (
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "special_tokens_map.json", "added_tokens.json", "tokenizer.model",
    "spiece.model", "sentencepiece.bpe.model", "vocab.txt",
    "preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json",
    "chat_template.jinja", "chat_template.json",
)


def _identity(path: Path) -> tuple[int, ...]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@lru_cache(maxsize=256)
def _file_digest(path: Path, identity: tuple[int, ...]) -> str:
    if not path.is_file():
        raise ValueError("本地模型制品不是文件")
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    if _identity(path) != identity:
        raise ValueError("本地模型制品在计算版本时发生变化")
    return digest.hexdigest()


def _manifest(root: Path, names: list[str], prefix: str = "") -> str:
    return "".join(f"{_file_digest(root / name, _identity(root / name))}  {prefix}{name}\n"
                   for name in names)


def _weight_files(root: Path) -> list[str]:
    for weights in ("model.safetensors", "pytorch_model.bin"):
        if (root / weights).is_file():
            return ["config.json", weights]
        index = f"{weights}.index.json"
        if (root / index).is_file():
            metadata = json.loads((root / index).read_text(encoding="utf-8"))
            weight_map = metadata.get("weight_map") if isinstance(metadata, dict) else None
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("本地模型权重索引无效")
            shards = list(weight_map.values())
            if not all(isinstance(name, str) and Path(name).name == name
                       and name.endswith(Path(weights).suffix) for name in shards):
                raise ValueError("本地模型权重分片路径无效")
            return ["config.json", index, *sorted(set(shards))]
    raise ValueError("本地模型缺少支持的权重文件")


def _processor_files(root: Path) -> list[str]:
    names = [name for name in _PROCESSOR_FILES if (root / name).is_file()]
    if not names:
        raise ValueError("本地模型缺少预处理制品")
    return names


def local_embedding_revisions(model_path: str, *, visual: bool = False) -> dict[str, str]:
    """从实际权重与预处理文件生成两枚可复现的 SHA-256 指纹。"""
    try:
        root = Path(model_path).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError("本地模型路径不是目录")
        with _LOCK:
            if visual:
                from ..visual.model.colqwen2_local import ColQwen2Embedding
                adapter, base = ColQwen2Embedding._resolve_paths(str(root))
                model = _manifest(base, _weight_files(base), "base/") + _manifest(
                    adapter, ["adapter_config.json", "adapter_model.safetensors"], "adapter/")
                processor = _manifest(adapter, _processor_files(adapter), "adapter/")
            else:
                code = sorted(path.name for path in root.glob("*.py") if path.is_file())
                model = _manifest(root, _weight_files(root) + code)
                processor = _manifest(root, _processor_files(root) + code)
    except (OSError, RuntimeError) as exc:
        raise ValueError("本地模型制品不可读，无法计算版本") from exc
    return {"model_revision": "sha256:" + sha256(model.encode()).hexdigest(),
            "processor_revision": "sha256:" + sha256(processor.encode()).hexdigest()}


async def with_local_embedding_metadata(data: dict, *, input_template) -> dict:
    category, provider = data.get("category"), data.get("provider")
    text = category == ModelCategory.TXT_EMBEDDING and provider in {"local_qwen", "local_bge"}
    visual = category == ModelCategory.IMG_EMBEDDING and provider == "local_qwen"
    if not (text or visual):
        return data
    path = data.get("model_path")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("模型未配置本地制品路径")
    revisions = await asyncio.to_thread(local_embedding_revisions, path, visual=visual)
    params = {**(data.get("model_params") or {}), **revisions}
    params.pop("input_template", None)
    normalized = {**data, "model_path": str(Path(path).expanduser().resolve()), "model_params": params}
    # 目录里的模板值不能覆盖运行时生效的规则。
    normalized["model_params"]["input_template"] = input_template(normalized)
    return normalized
