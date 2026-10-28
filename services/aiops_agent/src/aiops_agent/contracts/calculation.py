"""有界公式依赖图合同；输入和语义由受控观测生产者提供。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aiops_agent.contracts.measurements import ScalarMeasurement


class CalculationStep(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    operator: Literal["COUNTER_DELTA", "RATE", "DIVIDE", "SUM"]
    inputs: tuple[str, ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def validate_arity(self):
        required = {"COUNTER_DELTA": 2, "RATE": 1, "DIVIDE": 2}
        if self.operator in required and len(self.inputs) != required[self.operator]:
            raise ValueError("公式输入数量不符")
        if len(set(self.inputs)) != len(self.inputs):
            raise ValueError("同一公式不能重复计入同一输入")
        return self


class CalculationGraph(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["MEASUREMENT_GRAPH.v1"] = "MEASUREMENT_GRAPH.v1"
    formula_version: Literal["measurement-algebra.v1"] = "measurement-algebra.v1"
    roots: dict[str, ScalarMeasurement] = Field(min_length=1, max_length=256)
    steps: tuple[CalculationStep, ...] = Field(min_length=1, max_length=128)
    outputs: tuple[str, ...] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def validate_dependencies(self):
        nodes = {step.node_id: step for step in self.steps}
        if len(nodes) != len(self.steps) or set(nodes) & set(self.roots):
            raise ValueError("公式节点身份不能重复或覆盖原始输入")
        all_ids = set(self.roots) | set(nodes)
        if any(not item or len(item) > 64 for item in all_ids):
            raise ValueError("公式引用无效")
        if any(not set(step.inputs) <= all_ids for step in self.steps):
            raise ValueError("公式引用不存在的输入")
        if len(set(self.outputs)) != len(self.outputs) or not set(self.outputs) <= nodes.keys():
            raise ValueError("输出必须唯一引用计算节点")
        states, depths = {}, {}
        def visit(key):
            if key in self.roots:
                return 0
            if states.get(key) == "ACTIVE":
                raise ValueError("公式依赖不能成环")
            if key in depths:
                return depths[key]
            states[key] = "ACTIVE"
            depth = 1 + max(visit(item) for item in nodes[key].inputs)
            if depth > 32:
                raise ValueError("公式依赖深度超出预算")
            states[key] = "DONE"
            depths[key] = depth
            return depth
        for key in nodes:
            visit(key)
        return self


class CalculationValue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    measurement: ScalarMeasurement
    quality: Literal["GOOD", "MISSING", "INCOMPARABLE"]
    reasons: tuple[str, ...] = ()
    leaf_refs: tuple[str, ...]
    native_origin_ids: tuple[str, ...]


class CalculationEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    graph: CalculationGraph
    values: dict[str, CalculationValue]
    decimal_precision: int
    rounding: Literal["ROUND_HALF_EVEN"] = "ROUND_HALF_EVEN"
