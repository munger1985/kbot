"""只接受已解析 Action Template 的确定性 Renderer。"""

from __future__ import annotations

import hashlib
import json

from .contracts import RenderedAction, ResolvedActionTemplate
from .validation import validate_rendered_action


def _sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


class ActionRenderer:
    def render(
        self,
        template: ResolvedActionTemplate,
        parameters: dict[str, object],
    ) -> RenderedAction:
        definition = template.definition
        expected = {item.name: item for item in definition.parameters}
        if set(parameters) != set(expected):
            raise ValueError("Action 参数集合与模板不一致")
        normalized: dict[str, int | str] = {}
        for name, rule in expected.items():
            value = parameters[name]
            if rule.type == "integer":
                if isinstance(value, bool):
                    raise ValueError("Action 整数参数不能是布尔值")
                parsed = int(value)
                assert rule.minimum is not None
                assert rule.maximum is not None
                if not rule.minimum <= parsed <= rule.maximum:
                    raise ValueError("Action 整数参数超出允许范围")
                normalized[name] = parsed
            else:
                parsed = str(value)
                if parsed not in rule.enum:
                    raise ValueError("Action 枚举参数无效")
                normalized[name] = parsed
        command = template.command_template.strip()
        for name, value in normalized.items():
            command = command.replace(f"{{{{{name}}}}}", str(value))
        validate_rendered_action(command, db_type=definition.db_type)
        parameters_hash = _sha256(normalized)
        return RenderedAction(
            action_template_id=definition.action_template_id,
            action_template_version=definition.version,
            variant=definition.variant,
            db_type=definition.db_type,
            renderer_version=definition.renderer_version,
            typed_parameters=normalized,
            parameters_hash=parameters_hash,
            command_text=command,
            command_hash=hashlib.sha256(command.encode()).hexdigest(),
            template_hash=template.template_hash,
            risk_level=definition.risk_level,
            execution_capability=definition.execution_capability,
            precondition_tool_refs=definition.precondition_tool_refs,
            verification_tool_refs=definition.verification_tool_refs,
            expected_effects=definition.expected_effects,
            rollback_description=definition.rollback_description,
            statement_timeout_seconds=(
                definition.statement_timeout_seconds
            ),
            observation_delay_seconds=(
                definition.observation_delay_seconds
            ),
            idempotency_class=definition.idempotency_class,
            concurrency_key=definition.concurrency_key,
        )
