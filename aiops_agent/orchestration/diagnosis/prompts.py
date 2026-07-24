"""版本化 Prompt 资产加载与完整性校验。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PROMPT_ROOT = Path(__file__).resolve().parent / "prompt_assets"


@dataclass(frozen=True)
class PromptAsset:
    prompt_id: str
    version: str
    sha256: str
    content: str

    def ref(self) -> dict[str, str]:
        return {
            "prompt_id": self.prompt_id,
            "prompt_version": self.version,
            "prompt_sha256": self.sha256,
        }


class DiagnosisPromptRegistry:
    def __init__(self, assets: tuple[PromptAsset, ...]):
        self._assets = {
            (item.prompt_id, item.version): item for item in assets
        }
        if len(self._assets) != len(assets):
            raise ValueError("诊断 Prompt ID 与版本不能重复")

    @classmethod
    def load(
        cls, root: Path | None = None
    ) -> "DiagnosisPromptRegistry":
        prompt_root = root or DEFAULT_PROMPT_ROOT
        assets = []
        for path in sorted(prompt_root.glob("*.v*.txt")):
            stem = path.name.removesuffix(".txt")
            prompt_id, version = stem.rsplit(".v", 1)
            content = path.read_text(encoding="utf-8")
            assets.append(
                PromptAsset(
                    prompt_id=prompt_id,
                    version=version,
                    sha256=hashlib.sha256(content.encode()).hexdigest(),
                    content=content,
                )
            )
        if not assets:
            raise ValueError("诊断 Prompt 目录为空")
        return cls(tuple(assets))

    def resolve(self, prompt_id: str, version: str = "1") -> PromptAsset:
        try:
            return self._assets[(prompt_id, version)]
        except KeyError as exc:
            raise LookupError(
                f"诊断 Prompt 不存在：{prompt_id}@{version}"
            ) from exc

    @property
    def snapshot(self) -> tuple[dict[str, str], ...]:
        return tuple(
            item.ref()
            for item in sorted(
                self._assets.values(), key=lambda asset: asset.prompt_id
            )
        )
