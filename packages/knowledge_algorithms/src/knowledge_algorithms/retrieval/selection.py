"""确定性的目标覆盖与多维载荷预算选择；结果排名不受此阶段删改。"""

from dataclasses import dataclass
from enum import Enum

from ..contracts.errors import require
from ..contracts.identity import EvidenceKey, nonempty


@dataclass(frozen=True, slots=True)
class PayloadCost:
    tokens: int
    images: int
    byte_count: int

    def __post_init__(self) -> None:
        require(all(type(v) is int and v >= 0 for v in (
            self.tokens, self.images, self.byte_count)), "载荷成本必须为非负整数")

    def fits(self, budget: "PayloadCost") -> bool:
        return (self.tokens <= budget.tokens and self.images <= budget.images
                and self.byte_count <= budget.byte_count)


@dataclass(frozen=True, slots=True)
class PayloadUnit:
    key: EvidenceKey
    cost: PayloadCost


@dataclass(frozen=True, slots=True)
class EvidenceGroup:
    group_key: str
    anchor_key: EvidenceKey
    members: tuple[PayloadUnit, ...]
    relation_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        nonempty(self.group_key)
        keys = [member.key for member in self.members]
        require(self.anchor_key in keys and len(keys) == len(set(keys)), "组必须包含唯一主锚点及不重复附件")
        require(all(key.owner_binding_ref == self.anchor_key.owner_binding_ref for key in keys),
                "证据组不能包含其他 App 的对象")


class MatchStatus(str, Enum):
    MATCHED = "MATCHED"
    UNKNOWN = "UNKNOWN"
    CONFLICT = "CONFLICT"


@dataclass(frozen=True, slots=True)
class CoverageTarget:
    target_key: str
    kind: str
    required: bool
    provenance: str
    plan_version: str

    def __post_init__(self) -> None:
        nonempty(self.target_key, self.kind, self.provenance, self.plan_version)
        require(type(self.required) is bool, "目标必需性必须为布尔值")


@dataclass(frozen=True, slots=True)
class CandidateTargetMatch:
    target_key: str
    evidence_key: EvidenceKey
    status: MatchStatus
    basis: tuple[str, ...]
    provenance: str

    def __post_init__(self) -> None:
        nonempty(self.target_key, self.provenance, *self.basis)
        require(isinstance(self.status, MatchStatus), "目标匹配状态无效")
        require(self.status != MatchStatus.MATCHED or bool(self.basis), "已匹配目标必须有可追溯依据")


@dataclass(frozen=True, slots=True)
class TargetCoverage:
    target_key: str
    status: str


@dataclass(frozen=True, slots=True)
class SelectionResult:
    selected_group_keys: tuple[str, ...]
    selected_evidence_keys: tuple[EvidenceKey, ...]
    cost: PayloadCost
    omitted: tuple[tuple[str, str], ...]
    coverage: tuple[TargetCoverage, ...]
    coverage_search_truncated: bool


def select_evidence_groups(
    groups: tuple[EvidenceGroup, ...], *, targets: tuple[CoverageTarget, ...],
    matches: tuple[CandidateTargetMatch, ...], budget: PayloadCost,
    max_cover_states: int,
) -> SelectionResult:
    """有界回溯预留必需目标，再按既有名次补充，不承诺全局最优。"""
    require(type(max_cover_states) is int and max_cover_states > 0, "覆盖重选预算必须显式提供")
    require(len({g.group_key for g in groups}) == len(groups), "证据组键重复")
    require(len({t.target_key for t in targets}) == len(targets), "覆盖目标键重复")
    owners = {g.anchor_key.owner_binding_ref for g in groups}
    require(len(owners) <= 1, "选择不能混用 App")
    units: dict[EvidenceKey, PayloadCost] = {}
    for group in groups:
        for member in group.members:
            require(member.key not in units or units[member.key] == member.cost,
                    "共享载荷成本矛盾，必须重新材料化核对")
            units[member.key] = member.cost
    target_keys = {target.target_key for target in targets}
    mapping: dict[tuple[str, EvidenceKey], MatchStatus] = {}
    for match in matches:
        require(match.target_key in target_keys and match.evidence_key in units, "覆盖映射身份不存在")
        pair = (match.target_key, match.evidence_key)
        require(pair not in mapping or mapping[pair] == match.status, "同一目标映射状态矛盾")
        mapping[pair] = match.status
    group_targets = [frozenset(
        target for target in target_keys if any(
            mapping.get((target, member.key)) == MatchStatus.MATCHED for member in group.members)
    ) for group in groups]
    required = tuple(target.target_key for target in targets if target.required)

    def contents(selected: frozenset[int]) -> set[EvidenceKey]:
        return {member.key for i in selected for member in groups[i].members}

    def cost(selected: frozenset[int]) -> PayloadCost:
        keys = contents(selected)
        return PayloadCost(sum(units[k].tokens for k in keys),
                           sum(units[k].images for k in keys),
                           sum(units[k].byte_count for k in keys))

    def covered(selected: frozenset[int]) -> set[str]:
        return {target for i in selected for target in group_targets[i]}

    best: frozenset[int] = frozenset()
    states = 0
    truncated = False

    def preference(selected: frozenset[int]) -> tuple:
        return (-len(set(required) & covered(selected)), tuple(sorted(selected)))

    # 显式工作栈避免目标数触发 Python 递归深度，状态数由 Profile 限制。
    stack: list[tuple[int, frozenset[int]]] = [(0, frozenset())]
    while stack:
        if states >= max_cover_states:
            truncated = True
            break
        index, selected = stack.pop()
        states += 1
        if preference(selected) < preference(best):
            best = selected
        if index >= len(required):
            if len(set(required) & covered(selected)) == len(required):
                best = selected
                break
            continue
        target = required[index]
        if target in covered(selected):
            stack.append((index + 1, selected))
            continue
        stack.append((index + 1, selected))
        candidates = [i for i, coverage in enumerate(group_targets) if target in coverage]
        for i in reversed(candidates):
            candidate = selected | {i}
            if cost(candidate).fits(budget):
                stack.append((index + 1, candidate))

    selected = best
    for i in range(len(groups)):
        if cost(selected | {i}).fits(budget):
            selected = selected | {i}
    selected_keys = contents(selected)
    ordered_keys = tuple(dict.fromkeys(
        member.key for i, group in enumerate(groups) if i in selected for member in group.members))
    coverage = []
    for target in targets:
        if target.target_key in covered(selected):
            status = "MATCHED"
        elif any(target.target_key in coverage for coverage in group_targets):
            status = "BUDGET_OR_SEARCH_LIMIT"
        elif any(mapping.get((target.target_key, key)) == MatchStatus.UNKNOWN for key in units):
            status = "UNKNOWN"
        else:
            status = "NO_CONFIRMED_MATCH"
        coverage.append(TargetCoverage(target.target_key, status))
    require(bool(selected_keys) or not any(cost(frozenset({i})).fits(budget) for i in range(len(groups))),
            "存在可装入的合法单元却静默零选择", stage="selection")
    return SelectionResult(
        tuple(group.group_key for i, group in enumerate(groups) if i in selected), ordered_keys,
        cost(selected), tuple((g.group_key, "CONTEXT_BUDGET_INSUFFICIENT")
                              for i, g in enumerate(groups) if i not in selected),
        tuple(coverage), truncated)
