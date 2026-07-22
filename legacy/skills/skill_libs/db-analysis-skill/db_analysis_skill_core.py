# skills/skill_libs/db-analysis-skill/db_analysis_skill_core.py
"""
DBAnalysisSkill v3 — RCA 引擎 + HITL 人机协同。

架构升级:
  v2 (旧): 融合 Prometheus + 专家SQL + SOP → 直接流式输出诊断结论
  v3 (新): v2 的能力 + 数据充分性动态检查 + 挂起-恢复多轮交互

新增能力:
  - 非恢复模式: 前置 LLM 检查证据是否足够，不足则 yield WAIT_FOR_USER 并中断
  - 恢复模式: 跳过检查，注入 hitl_history（完整排查 Timeline）到分析 prompt
  - 多轮 Timeline: hitl_history 追加而非覆盖，LLM 看到完整排查历史
  - 用户错误容错: 识别 user_error 字段，LLM 自动生成替代 SQL
  - Token 控制: 充分性检查 max_tokens=500，规则前置减少 LLM 调用
"""

import json
import uuid
from typing import Any, AsyncGenerator
from loguru import logger

from skills import BaseSkill, SkillMeta, SkillDomain, SkillRunMode
from agent.common.ops_context import OpsContextMemory
from platform_core.dictionary import PacketType
from platform_clients import AIModelClient
from agent.prompt import default_prompt
from platform_core.config import get_prompt_config


class DBAnalysisSkill(BaseSkill):
    """
    AIOps 专属核心故障根因诊断大脑 (RCA Engine) v3:
    融合 Prometheus 监控时序数据 + 专家 SQL 诊断结果 + RAG 运维手册 + 用户补充数据,
    由 LLM 扮演资深 Principal DBA 进行确定性根因推理。
    支持多轮 HITL 人机协同交互。
    """
    meta = SkillMeta(
        name="db-analysis-skill",
        description="【数据库运维专有工具 v3】融合 Prometheus 监控、专家 SQL 诊断、运维手册与用户补充数据, 进行多轮 RCA 根因分析并支持 HITL 人机协同",
        domain=SkillDomain.OPS,
        run_mode=SkillRunMode.READ_ONLY,
    )

    def __init__(self):
        super().__init__()
        self.model_client = AIModelClient()

    async def run_stream(
        self, context: OpsContextMemory, **kwargs
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        故障根因诊断核心流 v3:
        1. 检测恢复模式
        2. 非恢复模式 → 数据充分性检查 → 不足则 yield WAIT_FOR_USER
        3. 融合多路数据 + hitl_history → 流式输出 RCA 报告
        """
        trace_id = context.get("trace_id")
        query_text = context["command_or_query"]
        instance_id = context["instance_id"]
        db_type = context["db_type"]
        environment = context["environment"]
        llm_model = context["llm_model"]

        metric_results = context.get("metric_results", [])
        monitor_results = context.get("monitor_results", [])
        doc_results = context.get("doc_results", [])

        # ---- HITL: 恢复模式检测 ----
        is_resuming = context.get("is_resuming", False)
        hitl_history: list[dict] = context.get("hitl_history", [])

        logger.info(
            f"[{trace_id}] DBAnalysisSkill v3 诊断大脑激活 | 实例: {instance_id} | 引擎: {db_type} "
            f"| 恢复模式: {is_resuming} | HITL轮次: {len(hitl_history)} "
            f"| 监控数据: {len(monitor_results)} 条 | 诊断数据: {len(metric_results)} 条 | 手册: {len(doc_results)} 篇"
        )

        # ---- 拦截决策 (硬编码规则，不消耗 LLM Token) ----
        # 规则 1: 非恢复模式且两路探针都为空 → 直接 insufficient
        if not is_resuming and not metric_results and not monitor_results:
            logger.warning(f"[{trace_id}] 诊断终止: 无任何监控或诊断数据。")
            request_id = str(uuid.uuid4())
            yield {
                "type": PacketType.WAIT_FOR_USER,
                "content": {
                    "request_id": request_id,
                    "reason": (
                        f"在 {environment} 环境的 {db_type} 实例上未采集到任何 Prometheus 监控数据 "
                        f"和数据库诊断数据。请检查: 1) Prometheus Server 是否可达 "
                        f"2) 数据库连接配置是否正确 3) 监控账号权限是否充足"
                    ),
                    "sql_to_run": "",
                    "expected_fields": [],
                    "suspended_by": "db-analysis-skill",
                }
            }
            return

        # ---- HITL: 数据充分性前置检查 (仅在非恢复模式下) ----
        if not is_resuming:
            from .sufficiency_checker import DataSufficiencyChecker
            _checker = DataSufficiencyChecker(self.model_client)
            sufficiency = await _checker.check(
                query_text=query_text,
                metric_results=metric_results,
                monitor_results=monitor_results,
                doc_results=doc_results,
                hitl_history=hitl_history,
                db_type=db_type,
                environment=environment,
                llm_model=llm_model,
            )

            if sufficiency.get("verdict") == "insufficient":
                suggested = sufficiency.get("suggested_tools", [])
                if suggested:
                    # 有建议工具 → 自动执行，静默补充数据
                    ops_db_executor = context.get("variables", {}).get("_ops_db_executor")
                    yield {
                        "type": PacketType.THOUGHT,
                        "content": f"🔍 数据不足，正在自动采集补充信息...\n"
                    }
                    extra = await self._run_diagnostic_tools(
                        db_type=db_type, instance_id=instance_id,
                        tool_names=suggested, db_executor=ops_db_executor,
                    )
                    if extra:
                        metric_results = list(metric_results) + extra
                else:
                    # 无建议工具 → 不打扰用户，继续用现有数据分析
                    logger.info(f"[{trace_id}] 数据不足但无建议工具, 继续分析")

        # ---- 构建融合证据链 ----
        knowledge_context = self._build_knowledge_context(doc_results)
        monitor_context = self._build_monitor_context(monitor_results)
        metric_context = self._build_metric_context(metric_results)

        # ---- HITL: 构建多轮排查 Timeline 上下文 ----
        hitl_context = self._build_hitl_context(hitl_history)

        # ---- 渲染最终诊断 Prompt ----
        from datetime import datetime
        system_prompt = await default_prompt.generate(
            get_prompt_config().ops_diagnosis,
            environment=environment,
            db_type=db_type,
            version_code=context.get("version_code", 0),
            db_role=context.get("db_role", "primary"),
            current_time=context.get("variables", {}).get("client_time") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            variables=json.dumps(
                {k: v for k, v in context.get("variables", {}).items() if not k.startswith("_")},
                ensure_ascii=False,
            ),
            metric_results=json.dumps(metric_results, ensure_ascii=False, indent=2),
            monitor_results=monitor_context,
            os_log_snapshots=json.dumps(context.get("os_log_snapshots", []), ensure_ascii=False, indent=2),
            knowledge_context=knowledge_context,
            standalone_query=query_text,
            # HITL 新增: 注入多轮排查历史
            hitl_context=hitl_context,
        )

        logger.debug(
            f"[{trace_id}] 注入 LLM 的 prompt 数据: "
            f"monitor_lines={len([p for p in monitor_context.split(chr(10)) if p.strip()])}, "
            f"metric_lines={len([p for p in metric_context.split(chr(10)) if p.strip()])}, "
            f"hitl_context_len={len(hitl_context)}"
        )

        yield {"type": PacketType.THOUGHT, "content": "正在召集 DBA 专家大脑融合多路数据, 进行内核级 RCA 根因推演...\n"}

        # ---- 流式输出 RCA 报告 ----
        is_thinking = False
        output_buffer = ""
        try:
            async for chunk in self.model_client.get_llm_stream_parsed(
                model_name=llm_model,
                prompt=[{"role": "user", "content": system_prompt}],
                temperature=0.1
            ):
                if not chunk:
                    continue

                # 原生推理字段支持 (DeepSeek-R1 等)
                if hasattr(chunk, "reasoning_content") and chunk.reasoning_content:
                    yield {"type": PacketType.THOUGHT, "content": chunk.reasoning_content}
                    continue

                if not chunk.content:
                    continue

                output_buffer += chunk.content

                while output_buffer:
                    if not is_thinking:
                        if "<thought>" in output_buffer:
                            parts = output_buffer.split("<thought>", 1)
                            if parts[0].strip():
                                yield {"type": PacketType.ANSWER, "content": parts[0]}
                            is_thinking = True
                            output_buffer = parts[1]
                        elif "<" in output_buffer and any(
                            tag.startswith(output_buffer.rsplit("<", 1)[1])
                            for tag in ["thought>", "/thought>"]
                        ):
                            break
                        else:
                            yield {"type": PacketType.ANSWER, "content": output_buffer}
                            output_buffer = ""
                    else:
                        if "</thought>" in output_buffer:
                            parts = output_buffer.split("</thought>", 1)
                            if parts[0].strip():
                                yield {"type": PacketType.THOUGHT, "content": parts[0]}
                            is_thinking = False
                            output_buffer = parts[1]
                        elif "<" in output_buffer and "/thought>".startswith(
                            output_buffer.rsplit("<", 1)[1]
                        ):
                            break
                        else:
                            yield {"type": PacketType.THOUGHT, "content": output_buffer}
                            output_buffer = ""

            # 处理末尾残留
            if output_buffer.strip():
                m_type = PacketType.THOUGHT if is_thinking else PacketType.ANSWER
                yield {"type": m_type, "content": output_buffer}

        except Exception as e:
            logger.error(f"DBAnalysisSkill v3 运行异常: {e}")
            yield {"type": PacketType.ERROR, "content": f"⚠️ 分析中断: {str(e)}\n"}

    # ==================================================================
    # 内部方法
    # ==================================================================

    async def _run_diagnostic_tools(
        self, db_type: str, instance_id: str,
        tool_names: list[str], db_executor: Any = None,
    ) -> list[dict[str, Any]]:
        """自动执行诊断工具，返回 metric_results 格式"""
        if not tool_names or not db_executor:
            return []
        try:
            from agent.common.diagnostic_tools import DatabaseDiagnosticTools
            tools = DatabaseDiagnosticTools(db_type=db_type, db_executor=db_executor, instance_id=instance_id)
            results = []
            for name in tool_names[:3]:
                method = getattr(tools, name, None)
                if not method:
                    continue
                try:
                    data = await method()
                    if data:
                        results.append({
                            "step_id": f"auto_{name}", "task_description": f"自动: {name}",
                            "data": data, "meta": {"source": "auto_diagnostic", "tool_name": name},
                        })
                except Exception:
                    pass
            return results
        except Exception:
            return []




    def _build_hitl_context(self, hitl_history: list[dict]) -> str:
        """将多轮 HITL 交互渲染为 LLM 可读的排查历史"""
        if not hitl_history:
            return "（这是第一轮排查，暂无人工补充数据）"

        parts = ["## 📋 多轮排查 Timeline"]

        for entry in hitl_history:
            round_num = entry.get("round", "?")
            reason = entry.get("reason", "")
            sql = entry.get("sql_to_run", "")
            user_error = entry.get("user_error")
            user_data = entry.get("user_data")

            parts.append(f"\n### 第 {round_num} 轮")
            parts.append(f"- **Agent 请求原因**: {reason}")

            if sql:
                parts.append(f"- **Agent 让用户执行的 SQL**:\n```sql\n{sql}\n```")

            if user_error:
                error_text = str(user_error)[:500]
                parts.append(
                    f"- **⚠️ 用户执行报错**: {error_text}\n"
                    f"  注意: 用户没有执行此 SQL 的权限或 SQL 语法不兼容，"
                    f"  请尝试替代方案或使用更通用的系统视图。"
                )
            elif user_data:
                data_str = json.dumps(user_data, ensure_ascii=False,
                                      default=str, indent=2)
                # Token 控制: 最多保留 2000 字符
                if len(data_str) > 2000:
                    data_str = data_str[:2000] + "\n... (数据已截断)"
                parts.append(f"- **用户回填数据**:\n```json\n{data_str}\n```")

        return "\n".join(parts)

    def _build_knowledge_context(self, doc_results: list[dict]) -> str:
        """构建 SOP 知识库上下文"""
        if not doc_results:
            return "当前无匹配的专家 SOP 手册, 请依赖通用运维指标经验进行分析。"
        return "\n".join(
            f"- 《{d.get('file_name', '未命名文档')}》: {d.get('text_content', '')}"
            for d in doc_results
        )

    def _build_monitor_context(self, monitor_results) -> str:
        """清洗 Prometheus 原始数据为 LLM 可读的逐行明细"""
        parts = []
        for mr in (monitor_results if isinstance(monitor_results, list) else []):
            task = mr.get("task_description", "?")
            meta = mr.get("meta", {})
            metric_code = meta.get("metric_code", "?")
            raw_data = mr.get("data", [])
            if isinstance(raw_data, list) and len(raw_data) > 0:
                lines = [f"  - [{metric_code}] {task} (共 {len(raw_data)} 条):"]
                for d in raw_data:
                    labels = d.get("labels", {})
                    meaningful = {k: v for k, v in labels.items()
                                  if not k.startswith("__")
                                  and k not in ("instance", "job", "database")}
                    label_str = ", ".join(f"{k}={v}" for k, v in meaningful.items())
                    val = d.get("value", "N/A")
                    pct = f"{float(val) * 100:.2f}%" if isinstance(val, (int, float)) else str(val)
                    lines.append(f"      {label_str}: {pct}")
                parts.append("\n".join(lines))
            else:
                parts.append(f"  - [{metric_code}] {task}: 无数据")
        return "\n".join(parts) if parts else "（无 Prometheus 监控数据）"

    def _build_metric_context(self, metric_results) -> str:
        """清洗诊断数据为 LLM 可读格式"""
        parts = []
        for mr in (metric_results if isinstance(metric_results, list) else []):
            task = mr.get("task_description", "?")
            meta = mr.get("meta", {})
            tool = meta.get("tool_name", meta.get("metric_code", "?"))
            desc = meta.get("description", "")
            raw_data = mr.get("data", [])
            if isinstance(raw_data, list) and len(raw_data) > 0:
                lines = [f"- [{tool}] {task}:"]
                for d in raw_data[:20]:
                    if isinstance(d, dict):
                        readable = {k: v for k, v in d.items() if not k.startswith("__")}
                        lines.append(f"    {readable}")
                    else:
                        lines.append(f"    {d}")
                parts.append("\n".join(lines))
            elif isinstance(raw_data, dict):
                parts.append(
                    f"- [{tool}] {task}: {desc}\n"
                    f"    {json.dumps(raw_data, ensure_ascii=False, default=str)}"
                )
            else:
                parts.append(
                    f"- [{tool}] {task}: {desc} | 结果: {str(raw_data)[:500]}"
                )
        return "\n".join(parts) if parts else "（无数据库诊断数据）"
