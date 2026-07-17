"""AIOps 执行报告生成器 — 将诊断+执行+验证结果整合为专业的 Markdown 运维报告。

报告包含:
  1. 概览 (实例/环境/耗时/状态)
  2. 问题诊断 (原始问题 + LLM 诊断结论)
  3. 执行动作 (SQL + 影响 + 风险等级)
  4. 效果验证 (指标 before/after + 健康检查)
  5. 回滚信息 (如有)
  6. 后续建议 (LLM 生成)
"""
from datetime import datetime, timezone

from loguru import logger

from agent.common.ops_verifier import VerifyResult, VerifyStatus
from utils.clients import AIModelClient


# LLM 生成后续建议的 Prompt
_RECOMMENDATIONS_PROMPT = """你是一个 {db_type} 数据库运维专家。请基于以下自愈执行结果，给出后续优化建议。

## 验证状态: {status}
## 执行的变更:
{actions}

## 指标变化:
{metrics}

## 健康检查:
{health}

请提供 2-3 条专业的后续建议，涵盖:
1. 短期: 是否需要人工复核或补充操作
2. 长期: 如何预防该问题再次发生
3. 监控: 建议配置哪些告警阈值

使用中文，每条建议 1-2 句话，Markdown 格式（以 "- " 开头的列表项）。"""


class OpsReporter:
    """AIOps 执行报告生成器。

    生成专业运维报告 (Markdown 格式)，并调用 LLM 生成后续优化建议。
    """

    def __init__(self):
        self.model_client = AIModelClient()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate_report(
        self,
        instance_name: str,
        db_type: str,
        environment: str,
        trigger_type: str,
        original_question: str,
        diagnosis_summary: str,
        executed_actions: list[dict],
        verify_result: VerifyResult | None,
        rollback_info: dict | None,
        total_duration: float,
        llm_model: str = "",
    ) -> str:
        """生成完整的 Markdown 执行报告。

        Args:
            instance_name: 实例名称 (展示用)
            db_type: Oracle / PostgreSQL / MySQL
            environment: prod / staging / dev
            trigger_type: manual / webhook / cron
            original_question: 用户原始问题或告警摘要
            diagnosis_summary: LLM 诊断结论
            executed_actions: 已执行的动作列表 [{sql, impact, risk_level, context}]
            verify_result: 验证结果 (纯诊断无变更时为 None)
            rollback_info: 回滚信息 {rollback_sql, executed: bool, result} (如有)
            total_duration: 总耗时 (秒)
            llm_model: 用于生成建议的 LLM 模型名

        Returns:
            Markdown 格式的完整报告
        """
        status = verify_result.status if verify_result else VerifyStatus.VERIFIED
        status_icon = self._status_icon(status)

        lines: list[str] = []
        lines.append("# 数据库自愈执行报告")
        lines.append("")

        # --- 1. 概览 ---
        lines.append("## 1. 概览")
        lines.append("")
        lines.append("| 项目 | 详情 |")
        lines.append("|------|------|")
        lines.append(f"| 目标实例 | {instance_name} ({db_type}) |")
        lines.append(f"| 环境 | {environment} |")
        lines.append(f"| 触发方式 | {trigger_type} |")
        lines.append(f"| 执行耗时 | {total_duration:.1f}s |")
        lines.append(f"| 验证状态 | {status_icon} {status.value} |")
        lines.append(f"| 报告时间 | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} |")
        lines.append("")

        # --- 2. 问题诊断 ---
        lines.append("## 2. 问题诊断")
        lines.append("")
        lines.append(f"> {original_question}")
        lines.append("")
        lines.append(diagnosis_summary or "（无诊断摘要）")
        lines.append("")

        # --- 3. 执行动作 ---
        lines.append("## 3. 执行动作")
        lines.append("")
        if executed_actions:
            lines.append("| # | SQL | 影响 | 风险等级 |")
            lines.append("|---|-----|------|---------|")
            for i, action in enumerate(executed_actions, start=1):
                sql = (action.get("sql") or action.get("action_sql") or "")[:100]
                impact = action.get("impact", "")[:80]
                risk = action.get("risk_level", "medium")
                lines.append(f"| {i} | `{sql}` | {impact} | {risk} |")
        else:
            lines.append("（本次诊断未执行变更操作）")
        lines.append("")

        # --- 4. 效果验证 ---
        lines.append("## 4. 效果验证")
        lines.append("")
        if verify_result:
            lines.append(f"**判定结果**: {status_icon} {status.value}")
            lines.append("")

            pre = verify_result.pre_snapshot
            post = verify_result.post_snapshot
            if pre and post:
                lines.append("### 4.1 指标变化")
                lines.append("")
                lines.append("| 指标 | 修复前 | 修复后 | 变化 |")
                lines.append("|------|--------|--------|------|")
                for name, pre_data in pre.items():
                    pre_v = pre_data.get("value", "?")
                    post_v = post.get(name, {}).get("value", "?")
                    if isinstance(pre_v, (int, float)) and isinstance(post_v, (int, float)):
                        diff = post_v - pre_v
                        sign = "+" if diff > 0 else ""
                        lines.append(f"| {name} | {pre_v} | {post_v} | {sign}{diff:.1f} |")
                    else:
                        lines.append(f"| {name} | {pre_v} | {post_v} | — |")
                lines.append("")

            health = verify_result.health_check_result
            if health:
                lines.append("### 4.2 健康检查")
                lines.append("")
                lines.append("| 检查项 | 状态 | 详情 |")
                lines.append("|--------|------|------|")
                for check_name, check_data in health.items():
                    ok = "✅" if check_data.get("ok") else "❌"
                    detail = str(check_data.get("detail", ""))[:120]
                    lines.append(f"| {check_name} | {ok} | {detail} |")
                lines.append("")
        else:
            lines.append("（本次为纯诊断，未执行变更，跳过验证）")
        lines.append("")

        # --- 5. 回滚信息 ---
        lines.append("## 5. 回滚信息")
        lines.append("")
        if rollback_info and rollback_info.get("rollback_sql"):
            executed = rollback_info.get("executed", False)
            result = rollback_info.get("result", "")
            lines.append(f"- **回滚 SQL**: `{rollback_info['rollback_sql'][:200]}`")
            lines.append(f"- **是否执行**: {'是' if executed else '否（仅作为应急方案记录）'}")
            if result:
                lines.append(f"- **执行结果**: {result}")
        else:
            lines.append("本次操作无需回滚或未生成回滚方案。")
        lines.append("")

        # --- 6. 后续建议 ---
        lines.append("## 6. 后续建议")
        lines.append("")
        if verify_result and executed_actions and llm_model:
            try:
                recs = await self.generate_recommendations(
                    verify_result=verify_result,
                    actions=executed_actions,
                    db_type=db_type,
                    llm_model=llm_model,
                )
                lines.append(recs)
            except Exception as e:
                logger.warning(f"[OpsReporter] LLM 建议生成失败: {e}")
                lines.append("- 建议人工复核本次操作效果")
                lines.append("- 定期检查相关监控指标，设置告警阈值")
        else:
            lines.append("- 建议人工复核本次操作效果")
            lines.append("- 定期检查相关监控指标，设置告警阈值")
        lines.append("")

        lines.append("---")
        lines.append("*本报告由 Nexus AIOps 自动生成*")

        return "\n".join(lines)

    async def generate_recommendations(
        self,
        verify_result: VerifyResult,
        actions: list[dict],
        db_type: str,
        llm_model: str,
    ) -> str:
        """调用 LLM 基于验证结果和数据库类型生成后续优化建议。

        Prompt 策略:
        - VERIFIED:  建议定期巡检 + 预防性配置
        - DEGRADED:  建议人工复核 + 进一步诊断方向
        - FAILED:    建议升级处理 + 手动介入步骤
        - Oracle:    提及 AWR / ADDM / SQL Profile
        - PostgreSQL: 提及 VACUUM / ANALYZE / pg_stat_statements
        - MySQL:     提及 InnoDB 调优 / 慢查询日志
        """
        metrics_text = verify_result.summary or "（无指标数据）"
        health_data = verify_result.health_check_result or {}
        health_text = "\n".join(
            f"- {k}: {'✅' if v.get('ok') else '❌'} {v.get('detail', '')}"
            for k, v in health_data.items()
        ) or "（无健康检查数据）"

        actions_text = "\n".join(
            f"- [{a.get('risk_level', '?')}] {a.get('sql', a.get('action_sql', ''))[:200]}"
            for a in actions
        ) or "（无变更操作）"

        prompt = _RECOMMENDATIONS_PROMPT.format(
            db_type=db_type,
            status=verify_result.status.value,
            actions=actions_text,
            metrics=metrics_text,
            health=health_text,
        )

        try:
            response = await self.model_client.get_llm_answer(
                model_name=llm_model,
                prompt=prompt,
                temperature=0.3,
                max_tokens=400,
            )
            return str(response).strip()
        except Exception as e:
            logger.error(f"[OpsReporter] 建议生成失败: {e}")
            return "- 建议人工复核本次操作效果\n- 定期检查相关监控指标，设置告警阈值"

    @staticmethod
    def _status_icon(status: VerifyStatus) -> str:
        """返回状态对应的 emoji 图标。"""
        icons = {
            VerifyStatus.VERIFIED: "✅",
            VerifyStatus.DEGRADED: "⚠️",
            VerifyStatus.FAILED: "❌",
        }
        return icons.get(status, "❓")
