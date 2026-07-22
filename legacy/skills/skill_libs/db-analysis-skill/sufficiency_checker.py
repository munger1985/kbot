# skills/skill_libs/db-analysis-skill/sufficiency_checker.py
"""
DataSufficiencyChecker — 数据充分性前置检查器

职责: 在 LLM 正式分析前，判断当前收集到的数据是否足以完成诊断任务。
如果不足，生成具体的 SQL 供用户执行以补充数据（触发 HITL 中断）。

设计原则:
  - 独立于 DBAnalysisSkill，可复用（未来可被 Planner 或其他 Skill 调用）
  - 纯数据决策，不做业务分析
"""

import json
from typing import Any
from loguru import logger

from platform_clients import AIModelClient
from agent.prompt import default_prompt
from platform_core.config import get_prompt_config


class DataSufficiencyChecker:
    """数据充分性前置检查器 — 在 LLM 分析前判断数据是否足够"""

    def __init__(self, model_client: AIModelClient | None = None):
        self.model_client = model_client or AIModelClient()

    async def check(
        self,
        *,
        query_text: str,
        metric_results: list[dict[str, Any]],
        monitor_results: list[dict[str, Any]],
        doc_results: list[dict[str, Any]],
        hitl_history: list[dict[str, Any]],
        db_type: str,
        environment: str,
        llm_model: str,
    ) -> dict[str, Any]:
        """
        检查数据是否足够完成诊断。

        Returns:
            {
                "verdict": "sufficient" | "insufficient",
                "reason": "判断理由",
                "sql_to_run": "需要用户执行的补充 SQL (仅 insufficient 时有值)",
                "expected_fields": ["field1", "field2", ...],
            }
        """
        prompt = await default_prompt.generate(
            get_prompt_config().ops_sufficiency_check,
            query_text=query_text,
            db_type=db_type,
            environment=environment,
            metric_results=json.dumps(metric_results, ensure_ascii=False, indent=2),
            monitor_results=json.dumps(monitor_results, ensure_ascii=False, indent=2),
            doc_results=json.dumps(doc_results, ensure_ascii=False, indent=2),
            hitl_history=json.dumps(hitl_history, ensure_ascii=False, indent=2),
        )

        try:
            result = await self.model_client.get_llm_json(
                model_name=llm_model,
                prompt=prompt,
                temperature=0,
                max_tokens=500,
            )
            verdict = str(result.get("verdict", "sufficient")).strip().lower()
            return {
                "verdict": verdict,
                "reason": result.get("reason", ""),
                "suggested_tools": result.get("suggested_tools", []),
                "expected_fields": result.get("expected_fields", []),
            }
        except Exception as e:
            logger.error(f"[SufficiencyChecker] 检查失败: {e}, 默认判定为 sufficient 继续执行")
            return {
                "verdict": "sufficient",
                "reason": f"检查器异常: {e}",
                "sql_to_run": "",
                "expected_fields": [],
            }
