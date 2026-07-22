# agent/planners/ops_planner.py

import json
from datetime import datetime
from loguru import logger
from typing import AsyncGenerator, Any, cast

from platform_clients import AIModelClient
from utils.monitor import UnifiedMetricRegistry
from platform_core.dictionary import PacketType
from platform_core.config import get_prompt_config
from platform_core.database import db_instance
from agent.prompt import default_prompt
from dao.repositories import FileRepository, MemoryRepository
from skills import SkillManager, SkillDomain
from agent.common import OpsContextMemory, ExecutionPlan
from agent.common.diagnostic_tools import DatabaseDiagnosticTools


class OpsTaskPlanner:
    """
    自愈控制面动态任务规划器 (Prometheus优先·专家SQL兜底·双轨注入版)
    """

    def __init__(
        self,
        skill_manager: SkillManager,
        doc_orchestrator,
        metric_registry: UnifiedMetricRegistry | None = None,
    ):
        self.model_client = AIModelClient()
        self.skill_manager = skill_manager
        self.doc_orchestrator = doc_orchestrator
        self.metric_registry = metric_registry or UnifiedMetricRegistry()

    @property
    def db_session(self):
        return db_instance().get_session()

    async def _get_recent_chat_history(self, user_id: str, session_id: str) -> str:
        """获取近期对话历史，用于运维改写时的多轮指代消解"""
        try:
            async with self.db_session as session:
                repo = MemoryRepository(session)
                recent_entries = await repo.get_recent_entries(session_id, limit=3)
            if not recent_entries:
                return "（无历史对话记录，这是此会话的第一轮提问）"

            lines = []
            for entry in reversed(recent_entries):
                ans = (entry.answer or "")[:200]
                lines.append(f"User: {entry.raw_question}\nAssistant: {ans}")
            return "\n\n".join(lines)
        except Exception as e:
            logger.warning(f"[OpsPlanner] 获取会话历史失败: {e}，降级为空历史")
            return "（历史记录暂时不可用）"

    async def generate_plan(self, ctx: OpsContextMemory) -> AsyncGenerator[dict[str, Any], None]:
        """
        根据前端已锁定的实例与原始问题，运行精准 RAG 检索 SOP 并生成线性运维执行计划
        """
        logger.info(
            f"[OpsPlanner] 激活运维规划大脑 | 目标实例: {ctx['instance_id']} ({ctx['db_type']}) | 指令: {ctx['command_or_query']}"
        )

        # 🟢 纯净化平移：砍掉原有的 `if not ctx.get("instance_id")` 模糊匹配与反向澄清选单的一整套逻辑。
        # 此时 ctx['db_type'] 已经在编排器拉取完资产元数据后精准就绪，Planner 直接开工。

        yield {"type": PacketType.THOUGHT, "content": "正在启动运维专属改写引擎并检索 DBA 专家知识库...\n"}

        # 1. 从 ctx 中安全提取拓扑快照
        topology_snapshot = {
            "instance_id": ctx["instance_id"],
            "db_type": ctx["db_type"],
            "version_code": ctx.get("version_code", 0),
            "environment": ctx.get("environment", "dev")
        }
        
        # 🎯 多轮上下文感知：拉取近期对话历史用于指代消解
        chat_history = await self._get_recent_chat_history(
            user_id=ctx.get("user_id", "unknown"),
            session_id=ctx["session_id"]
        )

        system_prompt = await default_prompt.generate(
            get_prompt_config().ops_rewrite,
            raw_question=ctx["command_or_query"],
            topology=json.dumps(topology_snapshot, ensure_ascii=False),
            variables=json.dumps(
                {k: v for k, v in ctx["variables"].items() if not k.startswith("_")},
                ensure_ascii=False,
            ),
            chat_history=chat_history
        )

        standalone_query = ctx["command_or_query"]
        search_keywords = ctx["command_or_query"]

        yield {"type": PacketType.THOUGHT, "content": "🔍 正在理解运维意图并提取关键实体...\n"}

        try:
            rewrite_res = await self.model_client.get_llm_json(ctx["llm_model"], system_prompt)
            standalone_query = rewrite_res.get("standalone_query", ctx["command_or_query"])
            search_keywords = rewrite_res.get("search_keywords", ctx["command_or_query"])
            
            # 强类型安全回填
            if rewrite_res.get("extracted_variables"):
                ctx["variables"].update(rewrite_res["extracted_variables"])
        except Exception as e:
            logger.error(f"[OpsPlanner] 运维语义改写解析异常: {e}，降级使用原始问题。")

        # 2. 复用问文编排流水线，检索匹配的标准化 SOP
        try:
            pipe_out = await self.doc_orchestrator.run_pipeline(
                agent_id=ctx["agent_id"],
                standalone_query=standalone_query,
                search_keywords=search_keywords,
                security_level=1,
                tags=["ops", "sop"]  # 🎯 运维领域隔离：仅检索带运维/SOP标签的知识库文档
            )
            
            # 3. 映射文件名元数据，并存入特化的 doc_results 缓存区
            enriched_refs = await self._enrich_results_with_metadata(pipe_out.get('kb_results', []))
            if enriched_refs:
                ctx["doc_results"] = enriched_refs

                yield {
                    "type": PacketType.DOC_RESULTS,
                    "content": enriched_refs
                }
            
            else:
                logger.warning(f"[OpsPlanner] 前置故障 SOP 检索未返回有效结果，降级使用空列表。")
                enriched_refs = []
        except Exception as rag_err:
            # 捕获你日志中报出的异常，并记录精确日志
            logger.error(f"[OpsPlanner] 前置故障 SOP 检索发生异常: {rag_err}")
            enriched_refs = []

        # 5. 动态召集当前系统中具备运维标签的所有可用原子 Skill
        skills_list_str = self.skill_manager.get_skill_list_for_planner(domain_filter=SkillDomain.OPS)
        sop_context_str = "\n---\n".join([f"《{d['file_name']}》:\n{d['text_content']}" for d in enriched_refs])

        # 5.1 注入 Prometheus 可用监控指标清单（按当前 db_type 过滤，供 LLM 规划监控查询步骤）
        prometheus_metrics_str = self.metric_registry.list_for_llm_prompt(
            monitor_type="prometheus",
            db_type=ctx["db_type"],
        )

        # 5.2 注入 16 个专家诊断工具清单（供 LLM 在需要深入诊断时做单选题）
        diagnostic_tools_str = DatabaseDiagnosticTools.get_tool_manifest()

        # 6. 渲染任务蓝图生成 Prompt（双轨注入：监控指标 + 诊断工具）
        final_prompt_content = await default_prompt.generate(
            get_prompt_config().ops_planner,
            skills_list=skills_list_str,
            standalone_query=standalone_query,
            intent_type="ops_diagnose",
            existing_variables=", ".join(f"{k}={v}" for k, v in ctx["variables"].items() if not k.startswith("_")),
            db_type=ctx["db_type"],
            environment=ctx["environment"],
            sop_context=sop_context_str or "当前无匹配的专家 SOP 手册，请依赖通用运维指标经验进行线性探测排查。",
            prometheus_metrics=prometheus_metrics_str,
            diagnostic_tools=diagnostic_tools_str,
        )

        # 7. 模型决策出强类型的 ExecutionPlan
        try:
            plan_data = await self.model_client.get_llm_json(
                model_name=ctx["llm_model"],
                prompt=[{"role": "system", "content": final_prompt_content}],
                temperature=0.1
            )

            # 对齐 TypedDict 的 runtime_plan 结构契约
            ctx["runtime_plan"] = {
                "thought": plan_data.get("thought", "正在基于标准运维 SOP 指导编排集群诊断逻辑..."),
                "steps": plan_data.get("steps", []),
                "final_goal": plan_data.get("final_goal", "拉取故障实例内核观测指标进行闭环根因分析"),
                "plan_type": "dynamic",
                "workflow_id": None,
                "inputs": {
                    "user_query": standalone_query,
                    "plan_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "plan_type": "dynamic",
                    "model_name": ctx["llm_model"],
                    "intent": "ops_diagnose",
                    "context_vars": list(ctx["variables"].keys()) if ctx.get("variables") else [],
                    "agent_id": ctx.get("agent_id"),
                    "workflow_name": None,
                    "workflow_id": None,
                }
            }
            yield {"type": PacketType.THOUGHT, "content": f"💡 故障自愈控制面编排完毕: {ctx['runtime_plan']['thought']}\n"}

        except Exception as plan_err:
            logger.error(f"[OpsPlanner] 大模型生成计划失败: {plan_err}，安全熔断注入兜底单步计划。")
            ctx["runtime_plan"] = cast(ExecutionPlan, {
                "thought": "智能规划引擎产生异常，强制降级为基础内核指标安全查验。",
                "final_goal": "回捞底层观测指标以防故障扩大",
                "plan_type": "dynamic",
                "workflow_id": None,
                "inputs": {
                    "user_query": standalone_query,
                    "plan_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "plan_type": "dynamic",
                    "model_name": ctx["llm_model"],
                    "intent": None,
                    "context_vars": [],
                    "agent_id": ctx.get("agent_id"),
                    "workflow_name": None,
                    "workflow_id": None,
                },
                "steps": [{
                    "step_id": 1,
                    "skill": "db-metric-skill",
                    "task_description": f"内核关联指标安全探测: {standalone_query}",
                    "output_var": "fallback_metric_data",
                    "condition": None
                }]
            })

    async def _enrich_results_with_metadata(self, kb_results: list) -> list[dict[str, Any]]:
        """文件名元数据映射富化"""
        if not kb_results:
            return []

        file_ids = [str(res.file_id) for res in kb_results if getattr(res, 'file_id', None)]
        unique_file_ids = list(set(file_ids))

        file_name_map = {}
        try:
            async with self.db_session as session:
                file_repo = FileRepository(session)
                file_name_map = await file_repo.get_names_by_ids(unique_file_ids)
            return [
                {
                    **(res.to_dict() if hasattr(res, 'to_dict') else dict(res)),
                    "file_name": file_name_map.get(str(getattr(res, 'file_id', '')), "未知运维规范文档")
                }
                for res in kb_results
            ]
        except Exception as e:
            logger.error(f"[OpsPlanner] 映射引用文档文件名映射逻辑失败: {e}")
            return []