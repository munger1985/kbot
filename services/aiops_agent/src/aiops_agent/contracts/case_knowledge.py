"""案例专属知识结构；来源定位不作为另一份当前现场证据。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CaseClaimSource(BaseModel):
    """精确到冻结调查版本和判断的只读来源。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str = Field(min_length=1, max_length=128)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_id: str = Field(min_length=1, max_length=128)
    record_id: str = Field(min_length=1, max_length=128)
    record_revision: int = Field(ge=1)
    claim_id: str = Field(pattern=r"^c[0-9]+$")
    evidence_refs: tuple[str, ...] = Field(default=(), max_length=24)
    counterevidence_refs: tuple[str, ...] = Field(default=(), max_length=24)


class CaseInvestigationStep(BaseModel):
    """可读检查结果，而非模型隐藏推理；只从冻结记录确定性投影。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step_id: str = Field(pattern=r"^check-[0-9a-f]{24}$")
    kind: Literal["FACT", "CAUSE", "RISK", "PREDICTION", "EXCLUDED", "INTERPRETATION"]
    subject: str = Field(min_length=1, max_length=1000)
    finding: str = Field(min_length=1, max_length=3000)
    level: Literal["OBSERVED", "SUPPORTED", "HYPOTHESIS", "UNKNOWN"]
    applicable_time: str = Field(min_length=1, max_length=1000)
    limitations: tuple[str, ...] = Field(default=(), max_length=12)
    source: CaseClaimSource

    @model_validator(mode="after")
    def validate_support(self):
        if self.level in {"OBSERVED", "SUPPORTED"} and not self.source.evidence_refs:
            raise ValueError("案例中的有效发现必须有原始观测引用")
        if self.level == "OBSERVED" and (self.kind != "FACT" or self.source.counterevidence_refs):
            raise ValueError("原因或存在反证的判断不能标记为直接观测")
        return self


class CaseActionEffect(BaseModel):
    """逐项措施的效果映射；不能把案例总结果复制给每一项措施。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action_ordinal: int = Field(ge=1)
    result: Literal[
        "NOT_VERIFIED", "RESOLVED", "IMPROVED", "NOT_ACHIEVED",
        "ADVERSE", "DEGRADED", "UNCHANGED", "INCONCLUSIVE",
    ]
    method: str = Field(min_length=1, max_length=128)
    summary: str = Field(min_length=1, max_length=3000)
    evidence_refs: tuple[str, ...] = Field(min_length=1, max_length=64)
    limitations: tuple[str, ...] = Field(default=(), max_length=32)


class CaseKnowledgeUnitLocator(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    unit_id: str = Field(min_length=1, max_length=128)
    kind: Literal["PROBLEM", "CONCLUSION", "INVESTIGATION", "ACTION_EFFECT", "VERIFICATION", "BOUNDARIES", "UNRESOLVED"]
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    related_unit_ids: tuple[str, ...] = Field(default=(), max_length=16)

    @model_validator(mode="after")
    def validate_range(self):
        if self.end <= self.start:
            raise ValueError("案例正文定位范围必须非空")
        return self


class CaseKnowledgeManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["OPS_CASE_KNOWLEDGE_PROJECTION.v1"] = "OPS_CASE_KNOWLEDGE_PROJECTION.v1"
    title: str = Field(min_length=1, max_length=256)
    body_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    profile_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    offset_unit: Literal["UNICODE_CHARACTER"] = "UNICODE_CHARACTER"
    units: tuple[CaseKnowledgeUnitLocator, ...] = Field(min_length=1, max_length=5000)

    @model_validator(mode="after")
    def validate_units(self):
        ids = {unit.unit_id for unit in self.units}
        if len(ids) != len(self.units) or any(
            not set(unit.related_unit_ids) <= ids for unit in self.units
        ) or any(left.end > right.start for left, right in zip(self.units, self.units[1:])):
            raise ValueError("案例语义单元必须唯一、互不重叠且关联存在")
        return self
