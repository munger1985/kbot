"""把发现工具结果确定性绑定回原目录工具，不猜测参数、不依赖模型复述。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

from platform_core.contracts.aiops.investigation import (
    InvestigationAction,
    InvestigationPlan,
)


DISCOVERY_CONTINUATION_REQUIRED = "DISCOVERY_CONTINUATION_REQUIRED"
DISCOVERY_BINDING_AMBIGUOUS = "DISCOVERY_BINDING_AMBIGUOUS"
PRODUCT_TIMEZONE = ZoneInfo("Asia/Shanghai")
NEAREST_WINDOW = timedelta(minutes=30)
_PARAM_NAME = re.compile(
    r"^(?:(?P<window>[a-z0-9]+?)_)?(?P<bound>begin|end)_(?P<column>[a-z0-9_]+)$"
)
_ISO_DATETIME = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})?"
)


@dataclass(frozen=True)
class DiscoveryBindingResult:
    plan: InvestigationPlan
    status: str
    unbound_action_ids: tuple[str, ...] = ()
    clarification_question: str | None = None


@dataclass(frozen=True)
class DiscoveryContinuationDecision:
    action: str
    binding: DiscoveryBindingResult | None = None
    continuation_actions: tuple[InvestigationAction, ...] = ()


def catalog_tool_cards(
    available_tools: tuple[dict, ...] | None = None,
) -> tuple[dict, ...]:
    """优先使用规划期工具卡片；缺失时回退到诊断目录的参数声明顺序。"""
    cards: dict[str, dict] = {}
    for item in available_tools or ():
        tool_id = str(item.get("tool_id") or "").strip()
        if tool_id and tool_id not in cards:
            cards[tool_id] = item
    if cards:
        return tuple(cards.values())
    from aiops_agent.diagnostics.registry import DiagnosticRegistry

    for item in DiagnosticRegistry.load().tools:
        definition = item.definition
        if definition.tool_id in cards:
            continue
        card = {
            "tool_id": definition.tool_id,
            "input": {
                parameter.name: parameter.model_dump(
                    mode="json",
                    exclude={"name"},
                    exclude_none=True,
                )
                for parameter in definition.parameters
            },
        }
        if definition.discovery_tool_id:
            card["discovery_tool_id"] = definition.discovery_tool_id
        cards[definition.tool_id] = card
    return tuple(cards.values())


def merge_prior_deferred_catalog_actions(
    *,
    plan: InvestigationPlan,
    prior_plan: dict | None,
    available_tools: tuple[dict, ...] = (),
) -> InvestigationPlan:
    """上一轮被延期的目录工具若被新计划丢掉，按 tool_id 与原始输入补回。"""
    if not prior_plan:
        return plan
    try:
        previous = InvestigationPlan.model_validate(prior_plan)
    except (TypeError, ValueError):
        return plan
    tool_index = _tool_index(available_tools)
    used_ids = {action.action_id for action in plan.actions}

    def next_action_id() -> str:
        index = 1
        while f"a{index}" in used_ids:
            index += 1
        action_id = f"a{index}"
        used_ids.add(action_id)
        return action_id

    restored: list[InvestigationAction] = []
    for prior in previous.actions:
        if not prior.deferred:
            continue
        tool = tool_index.get(prior.tool_id)
        if tool is None or not str(tool.get("discovery_tool_id") or "").strip():
            continue
        if any(
            action.tool_id == prior.tool_id
            and not action.deferred
            and not _unusable_required_parameters(action.input, tool)
            for action in plan.actions
        ):
            continue
        if any(
            action.tool_id == prior.tool_id and dict(action.input) == dict(prior.input)
            for action in plan.actions
        ):
            continue
        discovery_tool_id = str(tool.get("discovery_tool_id") or "").strip()
        discovery_id = next(
            (
                action.action_id
                for action in plan.actions
                if action.tool_id == discovery_tool_id
            ),
            None,
        )
        restored.append(
            prior.model_copy(
                update={
                    "action_id": next_action_id(),
                    "depends_on": ((discovery_id,) if discovery_id else ()),
                    "deferred": True,
                }
            )
        )
    if not restored:
        return plan
    return plan.model_copy(update={"actions": (*plan.actions, *restored)})


def bind_discovery_parameters(
    *,
    plan: InvestigationPlan,
    tool_results: tuple[dict, ...],
    available_tools: tuple[dict, ...] = (),
    question: str | None = None,
) -> DiscoveryBindingResult:
    """用发现结果把可唯一对应的目录参数写成合法整数，无法唯一对应则保持延期。"""
    tool_index = _tool_index(available_tools)
    candidates = [
        action
        for action in plan.actions
        if _needs_binding(action, tool_index)
    ]
    if not candidates:
        return DiscoveryBindingResult(plan=plan, status="UNCHANGED")

    updated = {action.action_id: action for action in plan.actions}
    unbound: list[str] = []
    statuses: list[str] = []
    clarification = None
    discovery_cache: dict[str, dict] = {}

    for action in candidates:
        tool = tool_index[action.tool_id]
        discovery_tool_id = str(tool.get("discovery_tool_id") or "").strip()
        if discovery_tool_id not in discovery_cache:
            discovery_cache[discovery_tool_id] = _discovery_observation(
                tool_results, discovery_tool_id
            )
        bound, status = _bind_action_from_input(
            action=action,
            tool=tool,
            observation=discovery_cache[discovery_tool_id],
            question=question,
        )
        statuses.append(status)
        updated[action.action_id] = bound
        if status != "BOUND":
            unbound.append(action.action_id)
            if status == "AMBIGUOUS" and clarification is None:
                clarification = (
                    "发现结果无法唯一对应所需目录标识，请确认要使用的起止标识。"
                )

    synthesized_status, clarification = _synthesize_from_siblings(
        updated=updated,
        candidates=candidates,
        tool_index=tool_index,
        discovery_cache=discovery_cache,
        clarification=clarification,
    )
    if synthesized_status:
        statuses.append(synthesized_status)
        unbound = [
            action.action_id
            for action in plan.actions
            if updated[action.action_id].deferred
            and _needs_binding(updated[action.action_id], tool_index)
        ]

    new_plan = plan.model_copy(
        update={"actions": tuple(updated[action.action_id] for action in plan.actions)}
    )
    bound_count = sum(1 for status in statuses if status == "BOUND")
    if bound_count:
        return DiscoveryBindingResult(
            plan=new_plan,
            status="BOUND",
            unbound_action_ids=tuple(unbound),
            clarification_question=clarification,
        )
    if "AMBIGUOUS" in statuses:
        return DiscoveryBindingResult(
            plan=new_plan,
            status="AMBIGUOUS",
            unbound_action_ids=tuple(unbound),
            clarification_question=clarification
            or "发现结果无法唯一对应所需目录标识，请确认要使用的起止标识。",
        )
    if "WAITING" in statuses:
        return DiscoveryBindingResult(
            plan=new_plan,
            status="WAITING",
            unbound_action_ids=tuple(unbound),
        )
    return DiscoveryBindingResult(
        plan=new_plan,
        status="UNBINDABLE",
        unbound_action_ids=tuple(unbound),
        clarification_question=clarification
        or "发现结果无法对应所需目录标识，请确认要使用的起止标识。",
    )


def decide_discovery_continuation(
    *,
    plan: InvestigationPlan | dict | None,
    tool_results: tuple[dict, ...],
    available_tools: tuple[dict, ...] = (),
    question: str | None = None,
) -> DiscoveryContinuationDecision:
    """发现结果可唯一绑定时继续取证；零绑定且歧义时才向用户确认。"""
    parsed = _parse_plan(plan)
    if parsed is None or not any(action.deferred for action in parsed.actions):
        return DiscoveryContinuationDecision(action="NONE")
    binding = bind_discovery_parameters(
        plan=parsed,
        tool_results=tool_results,
        available_tools=available_tools,
        question=question,
    )
    if binding.status == "BOUND":
        continuation = bound_continuation_actions(
            prior_plan=parsed,
            bound_plan=binding.plan,
        )
        if continuation:
            return DiscoveryContinuationDecision(
                action="CONTINUE",
                binding=binding,
                continuation_actions=continuation,
            )
        return DiscoveryContinuationDecision(action="NONE", binding=binding)
    if binding.status in {"AMBIGUOUS", "UNBINDABLE"}:
        return DiscoveryContinuationDecision(action="ASK_USER", binding=binding)
    return DiscoveryContinuationDecision(action="NONE", binding=binding)


def bound_continuation_actions(
    *,
    prior_plan: InvestigationPlan,
    bound_plan: InvestigationPlan,
) -> tuple[InvestigationAction, ...]:
    """只继续执行上一轮被延期、本轮已绑定的目录工具，避免重复跑发现工具。"""
    previously_deferred = {
        action.action_id for action in prior_plan.actions if action.deferred
    }
    selected = tuple(
        action
        for action in bound_plan.actions
        if action.action_id in previously_deferred and not action.deferred
    )
    kept_ids = {action.action_id for action in selected}
    return tuple(
        action.model_copy(
            update={
                "depends_on": tuple(
                    item for item in action.depends_on if item in kept_ids
                )
            }
        )
        for action in selected
    )


def prior_plan_has_deferred(plan: InvestigationPlan | dict | None) -> bool:
    parsed = _parse_plan(plan)
    return parsed is not None and any(action.deferred for action in parsed.actions)


def _parse_plan(plan: InvestigationPlan | dict | None) -> InvestigationPlan | None:
    if plan is None:
        return None
    if isinstance(plan, InvestigationPlan):
        return plan
    if not isinstance(plan, dict):
        return None
    try:
        return InvestigationPlan.model_validate(plan)
    except (TypeError, ValueError):
        return None


def _tool_index(available_tools: tuple[dict, ...] | None) -> dict[str, dict]:
    return {
        str(item.get("tool_id") or ""): item
        for item in catalog_tool_cards(available_tools)
        if item.get("tool_id")
    }


def _needs_binding(action: InvestigationAction, tool_index: dict[str, dict]) -> bool:
    tool = tool_index.get(action.tool_id)
    if tool is None or not str(tool.get("discovery_tool_id") or "").strip():
        return False
    if action.deferred:
        return True
    return bool(_unusable_required_parameters(action.input, tool))


def _bind_action_from_input(
    *,
    action: InvestigationAction,
    tool: dict,
    observation: dict,
    question: str | None = None,
) -> tuple[InvestigationAction, str]:
    params = _bound_parameters(tool)
    if not params:
        return action, "UNBINDABLE"
    new_input = _normalized_integer_input(action.input, tool)
    if _all_required_integers_legal(new_input, tool):
        return (
            action.model_copy(update={"input": new_input, "deferred": False}),
            "BOUND",
        )
    new_input = _fill_datetimes_from_question(
        action, params, new_input, extra_text=question
    )
    missing_datetimes = []
    for param in params:
        current = new_input.get(param["name"])
        if _is_legal_integer(current, param["spec"]):
            continue
        if _parse_datetime(current) is None:
            missing_datetimes.append(param["name"])
    if missing_datetimes:
        if _window_names(params) and not any(
            _parse_datetime(new_input.get(param["name"])) for param in params
        ):
            return action, "NEEDS_SIBLINGS"
        return action, "UNBINDABLE"
    if not observation.get("found"):
        return action, "WAITING"
    if observation.get("failed") or not observation.get("rows"):
        return action, "UNBINDABLE"

    matched_rows = []
    instance_hint = None
    for param in params:
        current = new_input.get(param["name"])
        if _is_legal_integer(current, param["spec"]):
            row = _row_by_identifier(
                observation["rows"], param["column"], current
            )
            if row is not None:
                matched_rows.append(row)
                instance_hint = _instance_number(row, instance_hint)
            new_input[param["name"]] = _coerce_integer(current)
            continue
        target = _parse_datetime(current)
        matched, status = _match_row(
            target=target,
            rows=observation["rows"],
            instance_number=instance_hint,
        )
        if status:
            return action, status
        identifier = _coerce_integer(matched.get(param["column"]))
        if identifier is None or not _is_legal_integer(identifier, param["spec"]):
            return action, "UNBINDABLE"
        new_input[param["name"]] = identifier
        matched_rows.append(matched)
        instance_hint = _instance_number(matched, instance_hint)
    if not _same_instance(matched_rows):
        return action, "AMBIGUOUS"
    return (
        action.model_copy(update={"input": new_input, "deferred": False}),
        "BOUND",
    )


def _synthesize_from_siblings(
    *,
    updated: dict[str, InvestigationAction],
    candidates: list[InvestigationAction],
    tool_index: dict[str, dict],
    discovery_cache: dict[str, dict],
    clarification: str | None,
) -> tuple[str | None, str | None]:
    status = None
    for action in candidates:
        current = updated[action.action_id]
        tool = tool_index.get(current.tool_id) or {}
        params = _bound_parameters(tool)
        window_names = _window_names(params)
        if not current.deferred or not window_names:
            continue
        if _all_required_integers_legal(current.input, tool):
            continue
        discovery_tool_id = str(tool.get("discovery_tool_id") or "").strip()
        siblings = []
        for item in updated.values():
            if item.action_id == current.action_id or item.deferred:
                continue
            other = tool_index.get(item.tool_id) or {}
            if str(other.get("discovery_tool_id") or "").strip() != discovery_tool_id:
                continue
            other_windows = _window_names(_bound_parameters(other))
            if other_windows:
                continue
            if _unusable_required_parameters(item.input, other):
                continue
            start = _sibling_window_start(
                action=item,
                tool=other,
                observation=discovery_cache.get(discovery_tool_id) or {},
            )
            if start is None:
                continue
            siblings.append((start, item, other))
        if len(siblings) < len(window_names):
            continue
        if len(siblings) > len(window_names):
            status = "AMBIGUOUS"
            clarification = clarification or (
                "发现结果无法唯一对应所需目录标识，请确认要使用的起止标识。"
            )
            continue
        siblings.sort(key=lambda item: item[0])
        if any(
            siblings[index][0] == siblings[index + 1][0]
            for index in range(len(siblings) - 1)
        ):
            status = "AMBIGUOUS"
            clarification = clarification or (
                "发现结果无法唯一对应所需目录标识，请确认要使用的起止标识。"
            )
            continue
        new_input = dict(current.input or {})
        for window_name, (_start, sibling, sibling_tool) in zip(
            window_names, siblings
        ):
            sibling_params = {
                (param["bound"], param["column"]): param["name"]
                for param in _bound_parameters(sibling_tool)
            }
            for param in params:
                if param["window"] != window_name:
                    continue
                source_name = sibling_params.get((param["bound"], param["column"]))
                if source_name is None or source_name not in sibling.input:
                    new_input = None
                    break
                new_input[param["name"]] = sibling.input[source_name]
            if new_input is None:
                break
        if new_input is None or _unusable_required_parameters(new_input, tool):
            continue
        updated[action.action_id] = current.model_copy(
            update={"input": new_input, "deferred": False}
        )
        status = "BOUND"
    return status, clarification


def _bound_parameters(tool: dict) -> tuple[dict, ...]:
    schema = tool.get("input") or {}
    if not isinstance(schema, dict):
        return ()
    parsed = []
    for name, spec in schema.items():
        match = _PARAM_NAME.fullmatch(str(name))
        if match is None:
            continue
        parsed.append(
            {
                "name": str(name),
                "window": match.group("window"),
                "bound": match.group("bound"),
                "column": match.group("column"),
                "spec": spec if isinstance(spec, dict) else {},
            }
        )
    return tuple(parsed)


def _window_names(params: tuple[dict, ...]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            str(item["window"])
            for item in params
            if item.get("window")
        )
    )


def _all_required_integers_legal(values: dict, tool: dict) -> bool:
    return not _unusable_required_parameters(values or {}, tool)


def _unusable_required_parameters(values: dict, tool: dict) -> tuple[str, ...]:
    provided = values or {}
    schema = tool.get("input") or {}
    unusable = []
    names = []
    if isinstance(schema, dict):
        for name, spec in schema.items():
            if not isinstance(spec, dict):
                continue
            if spec.get("required", True):
                names.append(str(name))
    for name in names:
        if name not in provided:
            unusable.append(name)
            continue
        spec = schema.get(name) if isinstance(schema, dict) else None
        if isinstance(spec, dict) and not _matches_declared_input_type(
            provided[name], spec
        ):
            unusable.append(name)
    return tuple(unusable)


def _matches_declared_input_type(value: object, spec: dict) -> bool:
    expected = str(spec.get("type") or "").strip().lower()
    if expected == "integer":
        return _is_legal_integer(value, spec)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    return True


def _is_legal_integer(value: object, spec: dict | None = None) -> bool:
    candidate = _coerce_integer(value)
    if candidate is None:
        return False
    spec = spec or {}
    minimum = spec.get("minimum")
    maximum = spec.get("maximum")
    if isinstance(minimum, int) and candidate < minimum:
        return False
    if isinstance(maximum, int) and candidate > maximum:
        return False
    return True


def _coerce_integer(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, Decimal):
        if value != value.to_integral_value():
            return None
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not re.fullmatch(r"[+-]?\d+", text):
            return None
        try:
            return int(text)
        except ValueError:
            return None
    return None


def _normalized_integer_input(values: dict | None, tool: dict) -> dict:
    normalized = dict(values or {})
    schema = tool.get("input") or {}
    if not isinstance(schema, dict):
        return normalized
    for name, spec in schema.items():
        if name not in normalized or not isinstance(spec, dict):
            continue
        if str(spec.get("type") or "").strip().lower() != "integer":
            continue
        coerced = _coerce_integer(normalized[name])
        if coerced is not None:
            normalized[name] = coerced
    return normalized


def _fill_datetimes_from_question(
    action: InvestigationAction,
    params: tuple[dict, ...],
    current_input: dict,
    extra_text: str | None = None,
) -> dict:
    missing = [
        param
        for param in params
        if not _is_legal_integer(current_input.get(param["name"]), param["spec"])
        and _parse_datetime(current_input.get(param["name"])) is None
    ]
    if not missing:
        return current_input
    hints = _iso_datetimes_in_text(action.question)
    if len(hints) != len(missing):
        hints = _iso_datetimes_in_text(extra_text)
    if len(hints) != len(missing):
        return current_input
    filled = dict(current_input)
    for param, hint in zip(missing, hints):
        filled[param["name"]] = hint
    return filled


def _iso_datetimes_in_text(text: str | None) -> tuple[str, ...]:
    found = []
    for match in _ISO_DATETIME.findall(text or ""):
        if _parse_datetime(match) is not None:
            found.append(match)
    return tuple(found)


def _discovery_observation(tool_results: tuple[dict, ...], discovery_tool_id: str) -> dict:
    found = False
    failed = False
    columns: list[dict] = []
    rows: list[dict] = []
    for artifact in tool_results:
        payload = artifact
        if isinstance(artifact, dict) and artifact.get("schema_version") == "DBA_TOOL_RESULT.v1":
            payload = artifact.get("payload") or artifact
        if not isinstance(payload, dict):
            continue
        outcomes = payload.get("tool_outcomes") or ()
        for outcome in outcomes:
            if not isinstance(outcome, dict):
                continue
            if str(outcome.get("tool_id") or "") != discovery_tool_id:
                continue
            found = True
            gap = outcome.get("gap")
            status = str(outcome.get("status") or "")
            if gap or status in {"FAILED", "GAP"}:
                failed = True
                continue
            observation = outcome.get("observation") or {}
            raw_columns = list(observation.get("columns") or [])
            raw_rows = list(observation.get("rows") or [])
            if not raw_columns or not raw_rows:
                continue
            columns = [
                item if isinstance(item, dict) else {"name": str(item)}
                for item in raw_columns
            ]
            names = [str(item.get("name") or "") for item in columns]
            for raw in raw_rows:
                if isinstance(raw, dict):
                    row = dict(raw)
                elif isinstance(raw, (list, tuple)):
                    row = {
                        name: raw[index]
                        for index, name in enumerate(names)
                        if index < len(raw)
                    }
                else:
                    continue
                identity = _row_identity_time(row, columns)
                if identity is not None:
                    row["_time"] = identity
                rows.append(row)
    return {
        "found": found,
        "failed": failed and not rows,
        "columns": columns,
        "rows": rows,
    }


def _row_identity_time(row: dict, columns: list[dict]) -> datetime | None:
    for name in ("end_time", "begin_time"):
        parsed = _parse_datetime(row.get(name))
        if parsed is not None:
            return parsed
    for column in columns:
        if str(column.get("logical_type") or "").upper() != "DATETIME":
            continue
        parsed = _parse_datetime(row.get(str(column.get("name") or "")))
        if parsed is not None:
            return parsed
    return None


def _parse_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=PRODUCT_TIMEZONE)
    return parsed


def _wall_clock(value: datetime) -> datetime:
    """按产品墙钟比较快照时间，忽略错误的 UTC 标签。"""
    return datetime(value.year, value.month, value.day, value.hour, value.minute)


def _match_row(
    *,
    target: datetime | None,
    rows: list[dict],
    instance_number: object = None,
) -> tuple[dict | None, str | None]:
    if target is None:
        return None, "UNBINDABLE"
    target_clock = _wall_clock(target)
    candidates = [
        row
        for row in rows
        if row.get("_time") is not None
        and (
            instance_number is None
            or _same_identifier(row.get("instance_number"), instance_number)
        )
    ]
    exact = [
        row
        for row in candidates
        if _wall_clock(row["_time"]) == target_clock
    ]
    if len(exact) == 1:
        return exact[0], None
    if len(exact) > 1:
        return None, "AMBIGUOUS"
    nearest = []
    for row in candidates:
        delta = abs(_wall_clock(row["_time"]) - target_clock)
        if delta <= NEAREST_WINDOW:
            nearest.append((delta, row))
    nearest.sort(key=lambda item: (item[0], _wall_clock(item[1]["_time"])))
    if not nearest:
        return None, "UNBINDABLE"
    if len(nearest) > 1 and nearest[0][0] == nearest[1][0]:
        return None, "AMBIGUOUS"
    return nearest[0][1], None


def _row_by_identifier(rows: list[dict], column: str, value: object) -> dict | None:
    for row in rows:
        if _same_identifier(row.get(column), value):
            return row
    return None


def _same_identifier(left: object, right: object) -> bool:
    left_int = _coerce_integer(left)
    right_int = _coerce_integer(right)
    if left_int is not None and right_int is not None:
        return left_int == right_int
    return left == right


def _instance_number(row: dict, current: object) -> object:
    if "instance_number" not in row:
        return current
    value = row.get("instance_number")
    return current if current is not None else value


def _same_instance(rows: list[dict]) -> bool:
    values = []
    for row in rows:
        if "instance_number" not in row:
            continue
        current = row.get("instance_number")
        coerced = _coerce_integer(current)
        values.append(coerced if coerced is not None else current)
    return len(set(values)) <= 1


def _sibling_window_start(
    *,
    action: InvestigationAction,
    tool: dict,
    observation: dict,
) -> datetime | None:
    params = [item for item in _bound_parameters(tool) if item["bound"] == "begin"]
    if not params:
        return None
    begin = params[0]
    value = (action.input or {}).get(begin["name"])
    parsed = _parse_datetime(value)
    if parsed is not None:
        return parsed
    if _is_legal_integer(value, begin["spec"]):
        row = _row_by_identifier(observation.get("rows") or [], begin["column"], value)
        if row is not None:
            return row.get("_time")
    return None
