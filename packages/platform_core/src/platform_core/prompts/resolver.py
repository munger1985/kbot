"""数据库 Prompt 解析、完整性校验和严格渲染。"""

from __future__ import annotations

import hashlib
import json
import string
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from .repository import PlatformPromptRepository


class PromptNotFoundError(LookupError):
    """数据库中不存在目标 Prompt。"""


class PromptIntegrityError(RuntimeError):
    """Prompt Hash、变量或版本完整性校验失败。"""


@dataclass(frozen=True)
class ResolvedPrompt:
    prompt_key: str
    version: str
    sha256: str
    content: str
    input_variables: tuple[str, ...]
    output_schema: str | None
    source: str
    prompt_version_id: UUID | None = None

    def ref(self) -> dict[str, str | None]:
        return {
            "prompt_key": self.prompt_key,
            "prompt_version": self.version,
            "prompt_sha256": self.sha256,
            "source": self.source,
            "output_schema": self.output_schema,
            "prompt_version_id": (
                str(self.prompt_version_id)
                if self.prompt_version_id is not None
                else None
            ),
        }


class StrictPromptRenderer:
    @staticmethod
    def render(prompt: ResolvedPrompt, values: dict[str, Any]) -> str:
        expected = set(prompt.input_variables)
        actual = set(values)
        if expected != actual:
            raise PromptIntegrityError(
                f"Prompt 变量不匹配：missing={sorted(expected - actual)}，"
                f"unknown={sorted(actual - expected)}"
            )
        normalized = {
            key: (
                value
                if isinstance(value, str)
                else json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                )
            )
            for key, value in values.items()
        }
        try:
            return string.Template(prompt.content).substitute(normalized)
        except (KeyError, ValueError) as exc:
            raise PromptIntegrityError("Prompt 严格渲染失败") from exc


class PromptResolver:
    def __init__(self, *, session_factory):
        self._session_factory = session_factory

    async def resolve(
        self,
        prompt_key: str,
        *,
        version: str | None = None,
        prompt_version_id: UUID | None = None,
    ) -> ResolvedPrompt:
        async with self._session_factory() as session:
            repository = PlatformPromptRepository(session)
            if prompt_version_id is not None:
                row = await repository.get_version_by_id(
                    prompt_version_id=prompt_version_id
                )
            elif version is not None:
                row = await repository.get_version(
                    prompt_key=prompt_key, version=version
                )
            else:
                row = await repository.get_active(prompt_key=prompt_key)
            if row is None:
                identity = (
                    str(prompt_version_id)
                    if prompt_version_id is not None
                    else version or "ACTIVE"
                )
                raise PromptNotFoundError(
                    f"数据库 Prompt 不存在：{prompt_key}@{identity}"
                )
            definition, prompt_version = row
            if definition.prompt_key != prompt_key:
                raise PromptIntegrityError(
                    "冻结 Prompt Version 与 Key 不匹配"
                )
            return self._from_database(definition, prompt_version)

    @staticmethod
    def _from_database(definition, version) -> ResolvedPrompt:
        content = str(version.content)
        actual_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if actual_hash != version.content_sha256:
            raise PromptIntegrityError(
                f"数据库 Prompt Hash 不匹配：{definition.prompt_key}"
            )
        variables = version.input_variables_json
        if not isinstance(variables, list) or not all(
            isinstance(value, str) for value in variables
        ):
            raise PromptIntegrityError(
                f"数据库 Prompt 变量无效：{definition.prompt_key}"
            )
        return ResolvedPrompt(
            prompt_key=definition.prompt_key,
            version=version.version,
            sha256=version.content_sha256,
            content=content,
            input_variables=tuple(variables),
            output_schema=version.output_schema_ref,
            source="DATABASE",
            prompt_version_id=version.prompt_version_id,
        )
