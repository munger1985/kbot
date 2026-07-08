# skills/skill_libs/ops-heal-skill/ops_heal_skill_core.py
"""
OpsHealSkill — 运维愈合执行引擎

闭环流程:
  1. 从 ctx 读取 DBAnalysisSkill 的诊断报告和 metric_results
  2. 愈合循环 (最多5轮):
     a. LLM 决策: 查询数据？执行 SQL？还是已完成？
     b. 查询: 执行只读 SELECT 获取真实数据
     c. 执行: 调用 mutation 模式执行变更 SQL
     d. 失败 → LLM 根据错误修正 → 重试
     e. 成功 → 继续下一轮或结束
  3. 输出每步进度 + 最终结果
"""

import json
from typing import Any, AsyncGenerator
from loguru import logger

from skills import BaseSkill, SkillMeta, SkillDomain, SkillRunMode
from agent.common.ops_context import OpsContextMemory
from core.dictionary import PacketType
from utils.clients import AIModelClient, OpsDBExecutor
from agent.prompt import default_prompt
from core.config import get_prompt_config


class OpsHealSkill(BaseSkill):
    meta = SkillMeta(
        name="ops-heal-skill",
        description="【运维愈合执行引擎】读取诊断报告，自动查询数据库获取真实数据，生成并执行变更 SQL，失败时自愈重试",
        domain=SkillDomain.OPS,
        run_mode=SkillRunMode.MUTATION,
    )

    def __init__(self):
        super().__init__()
        self.model_client = AIModelClient()
        self.db_executor = OpsDBExecutor()

    async def run_stream(
        self, context: OpsContextMemory, **kwargs
    ) -> AsyncGenerator[dict[str, Any], None]:
        trace_id = context.get("trace_id")
        db_type = context["db_type"]
        instance_id = context["instance_id"]
        environment = context["environment"]
        llm_model = context["llm_model"]
        variables = context.get("variables", {})

        is_mutation = variables.get("is_mutation_allowed", False)
        if not is_mutation:
            yield {"type": PacketType.ANSWER, "content": "✅ 当前未开启变更许可，跳过愈合执行。\n"}
            return

        # 收集诊断报告和已有数据
        diagnosis = self._collect_diagnosis(context)
        if not diagnosis:
            yield {"type": PacketType.ANSWER, "content": "✅ 无诊断报告，跳过愈合执行。\n"}
            return

        knowledge = self._collect_data(context)

        yield {"type": PacketType.CALL, "content": {"skill": "ops-heal-skill", "description": "启动自愈愈合引擎，分析诊断报告并制定执行计划"}}
        yield {"type": PacketType.THOUGHT, "content": "🩺 正在分析诊断报告，寻找需要执行的自愈操作...\n"}

        max_rounds = 5
        results: list[dict] = []

        for rnd in range(1, max_rounds + 1):
            logger.info(f"[{trace_id}] OpsHeal 第{rnd}轮 | 已执行{len(results)}个动作")

            # LLM 决策
            decision = await self._llm_decide(
                diagnosis=diagnosis,
                knowledge=knowledge,
                round_num=rnd,
                max_rounds=max_rounds,
                results=results,
                db_type=db_type,
                environment=environment,
                llm_model=llm_model,
            )

            action = decision.get("action", "done")
            reason = decision.get("reason", "")

            if action == "done":
                yield {"type": PacketType.ANSWER, "content": f"✅ **愈合完成**: {reason}\n"}
                break

            elif action == "query":
                sql = decision.get("sql", "").strip()
                if not sql or not sql.upper().lstrip().startswith("SELECT"):
                    logger.warning(f"[{trace_id}] 决策返回无效查询 SQL")
                    continue
                logger.info(f"[{trace_id}] OpsHeal 执行查询: {sql[:150]}")
                yield {"type": PacketType.CALL, "content": {"skill": "ops-heal-skill", "description": f"查询数据库获取真实数据 (第{rnd}/{max_rounds}轮): {reason}"}}
                yield {"type": PacketType.THOUGHT, "content": f"📝 **执行查询**:\n```sql\n{sql}\n```\n"}
                try:
                    data = await self.db_executor.execute_readonly_ops_sql(
                        instance_id=instance_id, sql=sql,
                    )
                    if isinstance(data, list) and data:
                        knowledge += f"\n--- 查询结果 (第{rnd}轮) ---\n"
                        knowledge += json.dumps(data[:30], ensure_ascii=False, default=str)
                        knowledge += f"\n(共 {len(data)} 行, 展示前30行)"
                        yield {"type": PacketType.ANSWER, "content": f"✅ 查询返回 **{len(data)}** 行数据\n"}
                        results.append({"action": "query", "sql": sql, "rows": len(data), "status": "ok"})
                    else:
                        knowledge += f"\n--- 查询结果 (第{rnd}轮): 无数据 ---\n"
                        yield {"type": PacketType.ANSWER, "content": "⚠️ 查询无结果\n"}
                        results.append({"action": "query", "sql": sql, "rows": 0, "status": "empty"})
                except Exception as e:
                    logger.error(f"[{trace_id}] 查询失败: {e}")
                    knowledge += f"\n--- 查询失败: {e} ---\n"
                    results.append({"action": "query", "sql": sql, "status": "error", "error": str(e)})

            elif action == "execute":
                sql = decision.get("sql", "").strip()
                if not sql:
                    continue
                impact = decision.get("impact", "")
                rollback = decision.get("rollback_sql", "")

                logger.info(f"[{trace_id}] OpsHeal 执行变更: {sql[:150]}")
                # 注入审批 UI 所需信息到 ctx
                ctx_vars = context.get("variables", {})
                ctx_vars["pending_action_sql"] = sql
                ctx_vars["pending_action_impact"] = impact
                ctx_vars["pending_action_rollback"] = rollback
                ctx_vars["pending_action_risk_level"] = "medium"
                yield {"type": PacketType.CALL, "content": {"skill": "ops-heal-skill", "description": f"执行自愈变更 (第{rnd}/{max_rounds}轮): {reason}"}}
                yield {"type": PacketType.THOUGHT, "content": f"📝 **待执行 SQL**:\n```sql\n{sql}\n```\n⚠️ 影响: {impact}\n🔄 回滚: {rollback}\n"}

                success = False
                error_msg = ""
                for attempt in range(1, 4):
                    try:
                        result = await self.db_executor.execute_mutation_ops_sql(
                            instance_id=instance_id, sql=sql,
                        )
                        if isinstance(result, dict) and result.get("status") == "error":
                            error_msg = result.get("error_message", "未知错误")
                            logger.error(f"[{trace_id}] 执行失败 (第{attempt}次): {error_msg}")
                            if attempt < 3:
                                corrected = await self._llm_correct(
                                    sql, error_msg, db_type, knowledge, llm_model
                                )
                                if corrected and corrected != sql:
                                    sql = corrected
                                    yield {"type": PacketType.THOUGHT, "content": f"🔄 自愈重试 ({attempt+1}/3)...\n"}
                                    continue
                            break
                        else:
                            success = True
                            break
                    except Exception as e:
                        error_msg = str(e)
                        logger.error(f"[{trace_id}] 执行异常 (第{attempt}次): {e}")
                        if attempt < 3:
                            corrected = await self._llm_correct(
                                sql, str(e), db_type, knowledge, llm_model
                            )
                            if corrected and corrected != sql:
                                sql = corrected
                                continue
                        break

                results.append({
                    "action": "execute", "sql": sql, "status": "ok" if success else "failed",
                    "error": error_msg if not success else None,
                })

                if success:
                    yield {"type": PacketType.ANSWER, "content": f"✅ **执行成功**\n\n```sql\n{sql}\n```\n\n⚠️ **影响**: {impact}\n\n🔄 **回滚**: {rollback}\n"}
                    # 清理 pending
                    ctx_vars.pop("pending_action_sql", None)
                    ctx_vars.pop("pending_action_impact", None)
                    ctx_vars.pop("pending_action_rollback", None)
                else:
                    yield {"type": PacketType.ERROR, "content": f"❌ **执行失败**\n\n错误: {error_msg}\n\n回滚方案: {rollback}\n"}
                    if attempt < 3:
                        yield {"type": PacketType.THOUGHT, "content": f"🔄 将在下一轮自动修正并重试...\n"}
                    knowledge += f"\n--- 执行失败: {error_msg} ---\n"

        else:
            # 超过最大轮次
            yield {"type": PacketType.ANSWER, "content": self._summary(results, max_rounds)}

        logger.info(f"[{trace_id}] OpsHeal 结束, 共{len(results)}个动作")

    # ==================================================================
    # 私有方法
    # ==================================================================

    def _collect_diagnosis(self, context: OpsContextMemory) -> str:
        """从 execution_history 收集 DBAnalysisSkill 的诊断报告"""
        for entry in context.get("execution_history", []):
            if entry.get("status") not in ("success", "suspended"):
                continue
            if "analysis" in (entry.get("skill") or "").lower():
                answer = entry.get("answer", "") or entry.get("output", "")
                if isinstance(answer, str) and len(answer) > 50:
                    return answer
                if isinstance(answer, dict):
                    return str(answer)
        return ""

    def _collect_data(self, context: OpsContextMemory) -> str:
        """收集已有数据作为 LLM 的知识库"""
        parts = []
        for key in ("monitor_results", "metric_results", "doc_results"):
            data = context.get(key, [])
            if data:
                if key == "doc_results":
                    # SOP 手册提取文本内容
                    texts = []
                    for d in data:
                        if isinstance(d, dict):
                            name = d.get("file_name", "") or d.get("title", "")
                            content = d.get("text_content", "") or d.get("content", "")
                            if content:
                                texts.append(f"《{name}》: {content[:1000]}")
                    if texts:
                        parts.append(f"\n--- 运维 SOP 手册 ({len(texts)}篇) ---")
                        parts.append("\n".join(texts))
                else:
                    parts.append(f"\n--- {key} ({len(data)}条) ---")
                    parts.append(json.dumps(data, ensure_ascii=False, default=str)[:5000])
        return "\n".join(parts)

    async def _llm_decide(
        self, diagnosis, knowledge, round_num, max_rounds, results, db_type, environment, llm_model,
    ) -> dict[str, Any]:
        """LLM 决策下一步: query / execute / done"""
        results_str = json.dumps(results, ensure_ascii=False)[:2000] if results else "（首次执行）"
        prompt = await default_prompt.generate(
            get_prompt_config().ops_heal_decision,
            diagnosis=diagnosis[:4000],
            knowledge=knowledge[:4000],
            round_num=round_num,
            max_rounds=max_rounds,
            results=results_str,
            db_type=db_type,
            environment=environment,
        )
        try:
            return await self.model_client.get_llm_json(
                model_name=llm_model, prompt=prompt, temperature=0, max_tokens=500,
            )
        except Exception as e:
            logger.error(f"[OpsHeal] 决策失败: {e}")
            return {"action": "done", "reason": f"决策异常: {e}"}

    async def _llm_correct(
        self, failed_sql, error_msg, db_type, knowledge, llm_model,
    ) -> str:
        """LLM 根据错误修正 SQL"""
        prompt = f"""你是 {db_type} 专家。修正下面失败的 SQL。

失败 SQL: {failed_sql}
错误: {error_msg}
数据库知识: {knowledge[:3000]}

输出 JSON:
{{"sql": "修正后的 SQL", "reason": "说明"}}

规则:
- 只输出 JSON
- SQL 参数必须来自上面数据库知识中的真实值
- 禁止编造 SID/路径/表名
- 禁止生成 SELECT 查询"""
        try:
            r = await self.model_client.get_llm_json(
                model_name=llm_model, prompt=prompt, temperature=0, max_tokens=300,
            )
            sql = (r.get("sql") or "").strip()
            if sql and sql != failed_sql and not sql.upper().lstrip().startswith("SELECT"):
                return sql
        except Exception:
            pass
        return ""

    def _summary(self, results: list[dict], max_rounds: int) -> str:
        ok = sum(1 for r in results if r.get("status") == "ok")
        fail = sum(1 for r in results if r.get("status") in ("failed", "error"))
        return (
            f"## 自愈执行报告\n\n"
            f"- 共执行 {len(results)} 个动作（{ok} 成功, {fail} 失败）\n"
            f"- 已进行 {max_rounds} 轮，达到上限\n\n"
            f"> 💡 请根据以上信息手动完成剩余操作。\n"
        )
