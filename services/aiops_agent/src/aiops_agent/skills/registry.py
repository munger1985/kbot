"""代码拥有的版本化 DBA Skill Registry。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from pydantic import ValidationError

from platform_core.contracts.aiops.skills import DbaSkillManifest


class SkillCatalogError(ValueError):
    """Skill Manifest 或目录违反确定性约束。"""


def canonical_hash(value: object) -> str:
    """计算可跨进程重放的规范 JSON Hash。"""
    payload = (
        value.model_dump(mode="json")
        if hasattr(value, "model_dump")
        else value
    )
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class DbaSkillRegistry:
    """注册不可变 Manifest，并生成整个发布目录的内容 Hash。"""

    def __init__(
        self,
        manifests: Iterable[DbaSkillManifest] = (),
        *,
        allowed_tool_ids: frozenset[str] | None = None,
    ) -> None:
        self._items: dict[tuple[str, str], DbaSkillManifest] = {}
        self._hashes: dict[tuple[str, str], str] = {}
        self._allowed_tool_ids = allowed_tool_ids
        for manifest in manifests:
            self.register(manifest)
        self.validate_references()

    @classmethod
    def load(
        cls,
        root: Path,
        *,
        allowed_tool_ids: frozenset[str] | None = None,
    ) -> "DbaSkillRegistry":
        registry = cls(allowed_tool_ids=allowed_tool_ids)
        for path in sorted(root.glob("**/manifest.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                registry.register(DbaSkillManifest.model_validate(payload))
            except (OSError, json.JSONDecodeError, ValidationError) as exc:
                raise SkillCatalogError(f"Skill Manifest 无效：{path}") from exc
        registry.validate_references()
        return registry

    def register(self, manifest: DbaSkillManifest) -> None:
        key = (manifest.skill_id, manifest.version)
        if key in self._items:
            raise SkillCatalogError(
                f"Skill ID 与版本不能重复：{manifest.skill_id}@{manifest.version}"
            )
        if self._allowed_tool_ids is not None:
            unknown = sorted(
                step.tool_id
                for step in manifest.tool_dag
                if step.tool_id not in self._allowed_tool_ids
            )
            if unknown:
                raise SkillCatalogError(
                    f"Skill 引用了目录外 Tool：{', '.join(unknown)}"
                )
        self._items[key] = manifest
        self._hashes[key] = canonical_hash(manifest)

    def validate_references(self) -> None:
        known_ids = {skill_id for skill_id, _ in self._items}
        for manifest in self._items.values():
            unknown = sorted(set(manifest.fallback_skills) - known_ids)
            if unknown:
                raise SkillCatalogError(
                    f"Skill {manifest.skill_id} 引用了不存在的 fallback："
                    f"{', '.join(unknown)}"
                )

    @property
    def catalog_hash(self) -> str:
        entries = [
            {
                "skill_id": skill_id,
                "version": version,
                "manifest_hash": self._hashes[(skill_id, version)],
            }
            for skill_id, version in sorted(self._items)
        ]
        return canonical_hash(entries)

    def resolve(self, skill_id: str, version: str) -> DbaSkillManifest:
        try:
            return self._items[(skill_id, version)]
        except KeyError as exc:
            raise SkillCatalogError(
                f"Skill 不存在：{skill_id}@{version}"
            ) from exc

    def latest(self, skill_id: str) -> DbaSkillManifest:
        candidates = [
            manifest
            for (candidate_id, _), manifest in self._items.items()
            if candidate_id == skill_id
        ]
        if not candidates:
            raise SkillCatalogError(f"Skill 不存在：{skill_id}")
        return max(candidates, key=lambda item: self._version_key(item.version))

    def manifest_hash(self, skill_id: str, version: str) -> str:
        self.resolve(skill_id, version)
        return self._hashes[(skill_id, version)]

    def manifests(self) -> tuple[DbaSkillManifest, ...]:
        return tuple(self._items[key] for key in sorted(self._items))

    @staticmethod
    def _version_key(version: str) -> tuple[int, int, int]:
        return tuple(int(value) for value in version.split("."))  # type: ignore[return-value]
