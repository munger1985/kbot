"""案例入索引前进行语义识别和逐字替换；模型不改写技术结论或来源。"""

import asyncio
import re
from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from aiops_agent.application.case_redaction import CaseRedactor
from aiops_agent.adapters.model_serving import AIOpsModelError
from aiops_agent.application.configuration.common import sha256_json
from aiops_agent.application.errors import state_conflict
from aiops_agent.contracts.knowledge import CaseRedaction, DiagnosisCaseProfile


class SensitiveSpan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    field_id: str = Field(min_length=1, max_length=512)
    text: str = Field(min_length=2, max_length=2000)
    category: Literal["PERSON", "CUSTOMER", "BUSINESS_ID", "ACCOUNT", "HOST", "PATH", "SQL_LITERAL", "SAMPLE_DATA", "SECRET"]


class SemanticRedactionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["OPS_CASE_SEMANTIC_REDACTION.v1"] = "OPS_CASE_SEMANTIC_REDACTION.v1"
    spans: tuple[SensitiveSpan, ...] = Field(default=(), max_length=256)


def shared_content_hash(title, profile):
    data = profile.model_dump(mode="json")
    data.pop("redaction", None)
    data.pop("source", None)
    return sha256_json({"title": title, "profile": data})


def _fields(title, profile):
    """只提供可共享叙述，不向模型发送内部来源链或原始凭据。"""
    data = profile.model_dump(mode="json")
    fields, paths = {"title": title}, {}
    def visit(value, path):
        if isinstance(value, dict):
            for key, item in value.items():
                if key not in {"source", "redaction", "evidence_refs", "before_refs", "after_refs", "guardrail_refs", "controlled_verification_ref", "effect_result_ref", "registration_ref"}:
                    visit(item, (*path, key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, (*path, index))
        elif isinstance(value, str) and value:
            identity = "f" + str(len(paths) + 1)
            fields[identity], paths[identity] = value, path
    visit(data, ())
    return data, fields, paths


def apply_semantic_redaction(*, title, profile, output, model_ref, prompt_ref):
    data, fields, paths = _fields(title, profile)
    mapping, counts = {}, {}
    for span in output.spans:
        if span.field_id not in fields or span.text not in fields[span.field_id]:
            raise ValueError("语义脱敏必须逐字定位原字段，不允许臆造敏感内容")
        if span.text not in mapping:
            counts[span.category] = counts.get(span.category, 0) + 1
            mapping[span.text] = f"<semantic_{span.category}_{counts[span.category]}>"
    # 同一个字面值全篇一致；重叠取最长匹配，避免短串覆盖后漏掉较长敏感值。
    pattern = re.compile("|".join(re.escape(text) for text in sorted(mapping, key=lambda text: (-len(text), text)))) if mapping else None
    for field_id, value in fields.items():
        if pattern is not None:
            value = pattern.sub(lambda match: mapping[match.group(0)], value)
        fields[field_id] = value
        if field_id != "title":
            current = data
            path = paths[field_id]
            for part in path[:-1]:
                current = current[part]
            current[path[-1]] = value
    revised = DiagnosisCaseProfile.model_validate(data)
    revised.redaction = CaseRedaction(
        policy_version="case-redaction.v2",
        status="REVIEW_REQUIRED",
        replacement_counts={**dict(profile.redaction.replacement_counts if profile.redaction else {}), **{f"semantic_{key}": value for key, value in counts.items()}},
        semantic_provenance={
            "status": "SCANNED",
            "model": model_ref,
            "prompt": prompt_ref,
            "input_hash": shared_content_hash(title, profile),
            "output_hash": shared_content_hash(fields["title"], revised),
            "span_count": len(output.spans),
        },
    )
    return fields["title"], revised


async def redact_case_for_index(*, title, profile, collection, structured_model, prompt_registry, model_catalog, model_scope, technical_name_field="model_name", source_preflight=None):
    """语义扫描不可用时阻断索引，不把规则脱敏误报为完整审阅。"""
    profile = CaseRedactor().profile(profile)
    title = CaseRedactor().text(title)
    if structured_model is None or prompt_registry is None or model_catalog is None:
        raise state_conflict("案例语义脱敏服务未配置，未向知识索引发送正文")
    try:
        model_id = str(collection["models"]["parser_llm"])
        from uuid import UUID
        definition = await model_catalog.get_model(UUID(model_id))
        technical_name = str(definition.get(technical_name_field) or "").strip()
        if not technical_name:
            raise ValueError("案例知识库未配置可用的语义审核模型")
        prompt = await prompt_registry.resolve("case_semantic_redaction")
        _, fields, _ = _fields(title, profile)
        if sum(len(value) for value in fields.values()) > 128_000:
            raise ValueError("案例内容超出语义审核预算，应拆分案例后重新提炼")
        model_ref = {"model_id": model_id, "technical_name": technical_name, "revision": model_id, **model_scope}
        digest = shared_content_hash(title, profile)
        if source_preflight is not None:
            await source_preflight()
        result = await asyncio.wait_for(structured_model.generate_structured(
            purpose="operations_knowledge.case_semantic_redaction",
            output_model=SemanticRedactionOutput,
            model_snapshot=model_ref,
            prompt_ref={**prompt.ref(), "content": prompt.content},
            input_payload={"fields": [{"field_id": key, "content": value} for key, value in fields.items()]},
            deadline=datetime.now(UTC) + timedelta(seconds=60),
            idempotency_key=f"case-redaction:{digest}:{model_id}:{prompt.ref().get('prompt_sha256', '')}",
        ), timeout=60)
        output = SemanticRedactionOutput.model_validate(result.output)
        return apply_semantic_redaction(title=title, profile=profile, output=output, model_ref=model_ref, prompt_ref=prompt.ref())
    except (KeyError, TypeError, ValueError, LookupError, TimeoutError, AIOpsModelError):
        raise state_conflict("案例语义脱敏未完成或输出无法核验，未向知识索引发送正文") from None
