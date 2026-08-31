"""代码拥有的版本化 DBA Playbook Registry。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from pydantic import ValidationError

from platform_core.contracts.aiops.playbooks import DbaPlaybookManifest


DEFAULT_PLAYBOOK_CATALOG_ROOT = Path(__file__).resolve().parent / "catalog"


class PlaybookCatalogError(ValueError):
    """Playbook Manifest 或目录违反确定性约束。"""


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


class PlaybookRegistry:
    """注册不可变 Manifest，并生成整个发布目录的内容 Hash。"""

    def __init__(
        self,
        manifests: Iterable[DbaPlaybookManifest] = (),
        *,
        allowed_tools: frozenset[tuple[str, str]] | None = None,
    ) -> None:
        self._items: dict[tuple[str, str], DbaPlaybookManifest] = {}
        self._hashes: dict[tuple[str, str], str] = {}
        self._allowed_tools = allowed_tools
        for manifest in manifests:
            self.register(manifest)
        self.validate_references()

    @classmethod
    def load(
        cls,
        root: Path = DEFAULT_PLAYBOOK_CATALOG_ROOT,
        *,
        allowed_tools: frozenset[tuple[str, str]] | None = None,
    ) -> "PlaybookRegistry":
        registry = cls(allowed_tools=allowed_tools)
        for path in sorted(root.glob("**/manifest.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                registry.register(DbaPlaybookManifest.model_validate(payload))
            except (OSError, json.JSONDecodeError, ValidationError) as exc:
                raise PlaybookCatalogError(
                    f"Playbook Manifest 无效：{path}"
                ) from exc
        registry.validate_references()
        return registry

    def register(self, manifest: DbaPlaybookManifest) -> None:
        key = (manifest.playbook_id, manifest.version)
        if key in self._items:
            raise PlaybookCatalogError(
                f"Playbook ID 与版本不能重复：{manifest.playbook_id}@{manifest.version}"
            )
        if self._allowed_tools is not None:
            unknown = sorted(
                f"{step.tool_id}@{step.tool_version}"
                for step in manifest.tool_dag
                if (step.tool_id, step.tool_version)
                not in self._allowed_tools
            )
            if unknown:
                raise PlaybookCatalogError(
                    f"Playbook 引用了目录外 Tool：{', '.join(unknown)}"
                )
        self._items[key] = manifest
        self._hashes[key] = canonical_hash(manifest)

    def validate_references(self) -> None:
        known_ids = {playbook_id for playbook_id, _ in self._items}
        for manifest in self._items.values():
            unknown = sorted(set(manifest.fallback_playbooks) - known_ids)
            if unknown:
                raise PlaybookCatalogError(
                    f"Playbook {manifest.playbook_id} 引用了不存在的 fallback："
                    f"{', '.join(unknown)}"
                )

    @property
    def catalog_hash(self) -> str:
        entries = [
            {
                "playbook_id": playbook_id,
                "version": version,
                "manifest_hash": self._hashes[(playbook_id, version)],
            }
            for playbook_id, version in sorted(self._items)
        ]
        return canonical_hash(entries)

    def resolve(self, playbook_id: str, version: str) -> DbaPlaybookManifest:
        try:
            return self._items[(playbook_id, version)]
        except KeyError as exc:
            raise PlaybookCatalogError(
                f"Playbook 不存在：{playbook_id}@{version}"
            ) from exc

    def latest(self, playbook_id: str) -> DbaPlaybookManifest:
        candidates = [
            manifest
            for (candidate_id, _), manifest in self._items.items()
            if candidate_id == playbook_id
        ]
        if not candidates:
            raise PlaybookCatalogError(f"Playbook 不存在：{playbook_id}")
        return max(candidates, key=lambda item: self._version_key(item.version))

    def manifest_hash(self, playbook_id: str, version: str) -> str:
        self.resolve(playbook_id, version)
        return self._hashes[(playbook_id, version)]

    def manifests(self) -> tuple[DbaPlaybookManifest, ...]:
        return tuple(self._items[key] for key in sorted(self._items))

    @staticmethod
    def _version_key(version: str) -> tuple[int, int, int]:
        parts = tuple(int(value) for value in version.split("."))
        if len(parts) != 3:
            raise PlaybookCatalogError(
                f"Playbook版本必须采用三段数字格式：{version}"
            )
        return parts
