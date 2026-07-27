"""单文件 Prompt Catalog 的严格加载和完整性校验。"""

from __future__ import annotations

import hashlib
import re
import string
import tomllib
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PROMPT_CATALOG = (
    Path(__file__).resolve().parents[1] / "resources" / "prompts.toml"
)
_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
_VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")


class PromptCatalogError(ValueError):
    """Prompt Catalog 内容无效。"""


@dataclass(frozen=True)
class PromptCatalogEntry:
    prompt_key: str
    owner_service: str
    version: str
    active: bool
    purpose: str
    input_variables: tuple[str, ...]
    output_schema: str | None
    content: str
    sha256: str

    def ref(self, *, source: str) -> dict[str, str | None]:
        return {
            "prompt_key": self.prompt_key,
            "prompt_version": self.version,
            "prompt_sha256": self.sha256,
            "source": source,
            "output_schema": self.output_schema,
        }


@dataclass(frozen=True)
class PromptCatalog:
    schema_version: str
    entries: tuple[PromptCatalogEntry, ...]
    catalog_sha256: str

    def active_for(
        self, prompt_key: str
    ) -> PromptCatalogEntry | None:
        return next(
            (
                item
                for item in self.entries
                if item.prompt_key == prompt_key and item.active
            ),
            None,
        )

    def exact(
        self, prompt_key: str, version: str
    ) -> PromptCatalogEntry | None:
        return next(
            (
                item
                for item in self.entries
                if item.prompt_key == prompt_key
                and item.version == version
            ),
            None,
        )

    def for_services(
        self, services: set[str]
    ) -> tuple[PromptCatalogEntry, ...]:
        return tuple(
            item
            for item in self.entries
            if item.owner_service == "platform"
            or item.owner_service in services
        )


def _canonical_content(value: str) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise PromptCatalogError("Prompt 正文不能为空")
    return normalized + "\n"


def _template_variables(content: str) -> tuple[str, ...]:
    result: list[str] = []
    try:
        for match in string.Template.pattern.finditer(content):
            if match.group("invalid") is not None:
                raise PromptCatalogError("Prompt 包含无效的模板占位符")
            name = match.group("named") or match.group("braced")
            if name and name not in result:
                result.append(name)
    except ValueError as exc:
        raise PromptCatalogError("Prompt 模板占位符无法解析") from exc
    return tuple(result)


def load_prompt_catalog(
    path: Path | None = None,
) -> PromptCatalog:
    catalog_path = path or DEFAULT_PROMPT_CATALOG
    if not catalog_path.is_file():
        raise PromptCatalogError(f"Prompt Catalog 不存在：{catalog_path}")
    raw_bytes = catalog_path.read_bytes()
    try:
        raw = tomllib.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise PromptCatalogError("Prompt Catalog 不是有效 UTF-8 TOML") from exc
    schema_version = str(raw.get("schema_version") or "")
    if schema_version != "kbot-prompt-catalog.v1":
        raise PromptCatalogError("Prompt Catalog schema_version 无效")
    entries: list[PromptCatalogEntry] = []
    seen: set[tuple[str, str]] = set()
    active_keys: set[str] = set()
    for index, item in enumerate(raw.get("prompts") or (), start=1):
        if not isinstance(item, dict):
            raise PromptCatalogError(f"第 {index} 个 Prompt 不是对象")
        key = str(item.get("prompt_key") or "")
        owner = str(item.get("owner_service") or "")
        version = str(item.get("version") or "")
        if not _KEY_PATTERN.fullmatch(key):
            raise PromptCatalogError(f"Prompt Key 无效：{key}")
        if not re.fullmatch(r"^[a-z][a-z0-9_]*$", owner):
            raise PromptCatalogError(f"Prompt Owner 无效：{owner}")
        if not _VERSION_PATTERN.fullmatch(version):
            raise PromptCatalogError(f"Prompt Version 无效：{version}")
        identity = (key, version)
        if identity in seen:
            raise PromptCatalogError(f"Prompt 版本重复：{key}@{version}")
        seen.add(identity)
        active = bool(item.get("active", False))
        if active and key in active_keys:
            raise PromptCatalogError(f"Prompt 存在多个 Active 版本：{key}")
        if active:
            active_keys.add(key)
        content = _canonical_content(str(item.get("content") or ""))
        declared_variables = tuple(
            str(value) for value in item.get("input_variables") or ()
        )
        if len(set(declared_variables)) != len(declared_variables):
            raise PromptCatalogError(f"Prompt 变量重复：{key}@{version}")
        actual_variables = _template_variables(content)
        if set(actual_variables) != set(declared_variables):
            raise PromptCatalogError(
                f"Prompt 变量声明不匹配：{key}@{version}，"
                f"声明={sorted(declared_variables)}，"
                f"正文={sorted(actual_variables)}"
            )
        entries.append(
            PromptCatalogEntry(
                prompt_key=key,
                owner_service=owner,
                version=version,
                active=active,
                purpose=str(item.get("purpose") or "").strip(),
                input_variables=declared_variables,
                output_schema=(
                    str(item.get("output_schema") or "").strip() or None
                ),
                content=content,
                sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            )
        )
    if not entries:
        raise PromptCatalogError("Prompt Catalog 不能为空")
    all_keys = {item.prompt_key for item in entries}
    missing_active = sorted(all_keys - active_keys)
    if missing_active:
        raise PromptCatalogError(
            f"Prompt 缺少 Active 版本：{missing_active}"
        )
    return PromptCatalog(
        schema_version=schema_version,
        entries=tuple(entries),
        catalog_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
