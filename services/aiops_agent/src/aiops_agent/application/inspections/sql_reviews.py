"""按严重性逐条分析所有问题 SQL，模型失败不吞掉检查事实。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from loguru import logger
from pydantic import ValidationError
from platform_core.prompts import PromptIntegrityError

from aiops_agent.adapters.model_serving import AIOpsModelError
from aiops_agent.contracts.inspection_sql_review import InspectionSqlReviewDraft
from .sql_problems import PROBLEM_SQL_TOOL, classify_problem_sql, problem_sql_sort_key


async def review_inspection_sql(*, context, result, model, prompts, concurrency: int) -> tuple[dict[str, Any], ...]:
    inspection = dict(dict(context.plan_snapshot.get("client_metadata") or {}).get("inspection") or {})
    step = next((step for step in inspection.get("evidence_steps") or ()
                 if step.get("tool_id") == PROBLEM_SQL_TOOL), None)
    if context.trigger_type != "SCHEDULE" or step is None or not step.get("sql_problem_policy"):
        return ()
    database_type = str(dict(context.plan_snapshot.get("target") or {}).get("db_type") or inspection.get("database_type") or "")
    problems = []
    for fact in result.evidence:
        if fact.tool_id != PROBLEM_SQL_TOOL or fact.trust_level != "SOURCE_VERIFIED":
            continue
        names = [column["name"] for column in fact.columns]
        for index, values in enumerate(fact.rows):
            if len(values) != len(names):
                continue
            row = dict(zip(names, values, strict=True))
            problem = classify_problem_sql(row, policy=step["sql_problem_policy"], database_type=database_type)
            if problem:
                problems.append({**problem, "evidence_refs": [fact.evidence_ref], "row_index": index})
    problems.sort(key=problem_sql_sort_key)
    if not problems:
        return ()
    answer = dict(context.plan_snapshot.get("answer_context") or {})
    semaphore = asyncio.Semaphore(concurrency)
    deadline = datetime.fromisoformat(context.deadline_at.replace("Z", "+00:00")) if context.deadline_at else None
    if deadline is not None and deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=UTC)

    async def one(problem):
        record = {"problem_id": problem["problem_id"], "status": "UNAVAILABLE",
                  "analysis": "Agent 初步分析尚未完成，保留已识别问题与规则复核建议。",
                  "evidence_refs": problem["evidence_refs"]}
        async with semaphore:
            try:
                if model is None or prompts is None:
                    raise ValueError("Agent 分析模型或 Prompt 未配置")
                if deadline is not None and datetime.now(UTC) >= deadline:
                    raise AIOpsModelError("MODEL_DEADLINE_EXCEEDED")
                frozen = dict(answer.get("prompts") or {})
                if "inspection_sql_review" not in frozen:
                    raise ValueError("本次运行未冻结问题 SQL 分析 Prompt")
                prompt = await prompts.resolve("inspection_sql_review", frozen_prompts=frozen)
                generated = await model.generate_structured(
                    purpose="aiops.inspection-sql-review", output_model=InspectionSqlReviewDraft,
                    model_snapshot=dict(answer["model"]), prompt_ref={**prompt.ref(), "content": prompt.content},
                    input_payload={"database_type": database_type, "problem": problem,
                                   "statistics_scope": "当前保留的累计统计，不是日报窗口增量；没有执行计划，不确认根因。"},
                    deadline=deadline, idempotency_key=f"inspection-sql:{context.run_id}:{problem['problem_id']}:{context.attempt}",
                )
                draft = InspectionSqlReviewDraft.model_validate(generated.output)
                if draft.problem_id != problem["problem_id"] or set(draft.evidence_refs) != set(problem["evidence_refs"]):
                    raise ValueError("Agent 分析的 SQL 身份或证据引用不匹配")
                fields = {key for key, value in problem["observation"].items()
                          if value is not None and (key.startswith("average_") or key in {
                              "execution_count", "total_elapsed_ms", "maximum_elapsed_ms",
                          })}
                if any(not set(cause.supporting_metrics) <= fields for cause in draft.possible_causes):
                    raise ValueError("Agent 引用了该 SQL 不具有的证据字段")
                record = {**draft.model_dump(mode="json"), "status": "COMPLETE",
                          "model_receipt": generated.receipt.model_dump(mode="json"), "prompt_ref": prompt.ref()}
            except (AIOpsModelError, PromptIntegrityError, ValidationError, ValueError, LookupError, TimeoutError) as exc:
                logger.warning("问题 SQL 逐条分析未完成：run_id={} problem_id={} error_type={}",
                               context.run_id, problem["problem_id"], type(exc).__name__)
                record["error_code"] = getattr(exc, "code", None) or "INSPECTION_SQL_REVIEW_UNAVAILABLE"
        return record

    return tuple(await asyncio.gather(*(one(problem) for problem in problems)))
