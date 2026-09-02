"""加载并精确解析随代码发布的 Action Catalog。"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .contracts import (
    ActionTemplateDefinition,
    ResolvedActionTemplate,
)
from .validation import validate_action_template


DEFAULT_ACTION_ROOT = Path(__file__).resolve().parent / "catalog"
DESTRUCTIVE_EFFECT_CLASSES = frozenset(
    {
        "DATA_DELETION",
        "OBJECT_DELETION",
        "RECOVERY_MATERIAL_DELETION",
        "STATE_REPLACEMENT",
        "ARBITRARY_MUTATION",
    }
)


class ActionRegistry:
    def __init__(self, templates: tuple[ResolvedActionTemplate, ...]):
        identities = [
            (
                item.definition.action_template_id,
                item.definition.version,
                item.definition.db_type,
                item.definition.variant,
            )
            for item in templates
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("Action Catalog 存在重复模板 Variant")
        self._templates = templates
        for item in templates:
            definition = item.definition
            if (
                definition.effect_class in DESTRUCTIVE_EFFECT_CLASSES
                and definition.execution_mode != "MANUAL_ONLY"
            ):
                raise ValueError("破坏性 Action 只能登记为 MANUAL_ONLY")
        self.catalog_hash = hashlib.sha256(
            json.dumps(
                [
                    {
                        **item.definition.model_dump(mode="json"),
                        "template_hash": item.template_hash,
                    }
                    for item in sorted(
                        templates,
                        key=lambda value: (
                            value.definition.db_type,
                            value.definition.action_template_id,
                            value.definition.version,
                            value.definition.variant,
                        ),
                    )
                ],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    @classmethod
    def load(cls, root: Path | None = None) -> "ActionRegistry":
        catalog_root = root or DEFAULT_ACTION_ROOT
        templates = []
        for manifest in sorted(catalog_root.glob("*/manifest.json")):
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            for raw in payload.get("actions", []):
                definition = ActionTemplateDefinition.model_validate(raw)
                command = None
                if definition.command_ref is not None:
                    command_path = (
                        manifest.parent / definition.command_ref
                    ).resolve()
                    if not command_path.is_relative_to(
                        manifest.parent.resolve()
                    ):
                        raise ValueError("Action 命令模板路径越界")
                    command_bytes = command_path.read_bytes()
                    if (
                        hashlib.sha256(command_bytes).hexdigest()
                        != definition.command_sha256
                    ):
                        raise ValueError("Action 命令模板 Hash 不匹配")
                    command = command_bytes.decode("utf-8")
                    validate_action_template(command, definition)
                template_hash = hashlib.sha256(
                    json.dumps(
                        definition.model_dump(mode="json"),
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                templates.append(
                    ResolvedActionTemplate(
                        definition=definition,
                        command_template=command,
                        template_hash=template_hash,
                    )
                )
        if not templates:
            raise ValueError("Action Catalog 为空")
        return cls(tuple(templates))

    @property
    def templates(self) -> tuple[ResolvedActionTemplate, ...]:
        return self._templates

    def resolve(
        self,
        *,
        action_template_id: str,
        version: str,
        db_type: str,
        db_version: str,
        capabilities: set[str],
        entitlements: set[str],
        environment: str,
    ) -> ResolvedActionTemplate:
        match = re.search(r"\d+", db_version)
        if match is None:
            raise ValueError("Action 数据库版本无法识别")
        major = int(match.group())
        candidates = [
            item
            for item in self._templates
            if item.definition.action_template_id == action_template_id
            and item.definition.version == version
            and item.definition.db_type == db_type
            and item.definition.status == "ACTIVE"
            and item.definition.supported_version_min <= major
            < item.definition.supported_version_max_exclusive
            and set(item.definition.required_capabilities) <= capabilities
            and set(item.definition.required_entitlements) <= entitlements
            and environment in item.definition.environment_allowlist
        ]
        if len(candidates) != 1:
            raise LookupError("Action Template 没有唯一且精确的 Variant")
        return candidates[0]

    def compatible(
        self,
        *,
        db_type: str,
        db_version: str,
        capabilities: set[str],
        entitlements: set[str],
        environment: str,
        include_planned: bool = True,
    ) -> tuple[ResolvedActionTemplate, ...]:
        """列出 Target 可见动作；计划项可见但永远不可执行。"""
        match = re.search(r"\d+", db_version)
        if match is None:
            return ()
        major = int(match.group())
        statuses = {"ACTIVE", "PLANNED"} if include_planned else {"ACTIVE"}
        return tuple(
            item
            for item in self._templates
            if item.definition.db_type == db_type
            and item.definition.status in statuses
            and item.definition.supported_version_min <= major
            < item.definition.supported_version_max_exclusive
            and set(item.definition.required_capabilities) <= capabilities
            and set(item.definition.required_entitlements) <= entitlements
            and environment in item.definition.environment_allowlist
        )

    def resolve_exact(
        self,
        *,
        action_template_id: str,
        version: str,
        db_type: str,
        variant: str,
        template_hash: str,
    ) -> ResolvedActionTemplate:
        candidates = [
            item
            for item in self._templates
            if item.definition.action_template_id == action_template_id
            and item.definition.version == version
            and item.definition.db_type == db_type
            and item.definition.variant == variant
            and item.template_hash == template_hash
            and item.definition.status == "ACTIVE"
        ]
        if len(candidates) != 1:
            raise LookupError("Action Catalog 与审批快照不匹配")
        return candidates[0]
