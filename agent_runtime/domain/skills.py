"""版本化 Skill Manifest 与固定实现注册表。"""

from collections.abc import Callable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .planning import ExecutionMode


class DataClassification(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class ArtifactDeclaration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_type: str = Field(min_length=1, max_length=64)
    schema_version: str = Field(min_length=1, max_length=64)


class SkillManifest(BaseModel):
    """启动时审核并注册的 Skill 静态能力声明。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    skill_id: str = Field(pattern=r"^[a-z][a-z0-9-]{0,127}$")
    version: str = Field(
        pattern=r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$",
    )
    owner: str = Field(min_length=1, max_length=128)
    specialist: str = Field(min_length=1, max_length=64)
    description: str = Field(min_length=1, max_length=1000)
    input_schema: str = Field(min_length=1, max_length=128)
    output_artifacts: tuple[ArtifactDeclaration, ...] = Field(min_length=1)
    permissions: tuple[str, ...] = ()
    execution_mode: ExecutionMode
    idempotent: bool
    timeout_seconds: int = Field(ge=1, le=3600)
    max_retries: int = Field(default=0, ge=0, le=10)
    data_classification: DataClassification
    external_dependencies: tuple[str, ...] = ()


class SkillRegistry:
    """只接受显式 Manifest 与实现映射，不扫描文件或动态反射。"""

    def __init__(self):
        self._entries: dict[
            tuple[str, str], tuple[SkillManifest, Callable[..., Any]]
        ] = {}

    def register(
        self,
        manifest: SkillManifest,
        implementation: Callable[..., Any],
    ) -> None:
        key = (manifest.skill_id, manifest.version)
        if key in self._entries:
            raise ValueError(
                f"Skill 已注册：{manifest.skill_id}@{manifest.version}"
            )
        self._entries[key] = (manifest, implementation)

    def contains(self, skill_id: str | None, version: str | None) -> bool:
        if not skill_id or not version:
            return False
        return (skill_id, version) in self._entries

    def resolve(
        self, skill_id: str, version: str,
    ) -> tuple[SkillManifest, Callable[..., Any]]:
        try:
            return self._entries[(skill_id, version)]
        except KeyError as exc:
            raise LookupError(
                f"Skill 未注册：{skill_id}@{version}"
            ) from exc
