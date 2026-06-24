# skills/skill_libs/db-analysis-skill/db_analysis_skill_core.py

import json
from typing import Any, AsyncGenerator
from loguru import logger

from skills import BaseSkill, SkillMeta, SkillDomain, SkillRunMode
from agent.common.ops_context import OpsContextMemory
from core.dictionary import PacketType
from utils.clients import AIModelClient
from agent.prompt import default_prompt
from core.config import get_prompt_config


class DBAnalysisSkill(BaseSkill):
    """
    AIOps 专属核心故障根因诊断大脑 (RCA Engine) v2:
    融合 Prometheus 监控时序数据 + 专家 SQL 诊断结果 + RAG 运维手册,
    由 LLM 扮演资深 Principal DBA 进行确定性根因推理。
    """
    meta = SkillMeta(
        name="db-analysis-skill",
        description="【数据库运维专有工具 v2】融合 Prometheus 监控时序数据、专家 SQL 诊断结果与运维手册, 进行全方位故障根因分析（RCA）并提供自愈建议",
        domain=SkillDomain.OPS,
        run_mode=SkillRunMode.READ_ONLY,
    )

    def __init__(self):
        super().__init__()
        self.model_client = AIModelClient()

    async def run_stream(self, context: OpsContextMemory, **kwargs) -> AsyncGenerator[dict[str, Any], None]:
        """
        故障根因诊断核心流 v2:
        融合 monitor_results（Prometheus）+ metric_results（专家 SQL）+ doc_results（运维手册）,
        流式吐出最终 RCA 报告。
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

        logger.info(
            f"[{trace_id}] DBAnalysisSkill v2 诊断大脑激活 | 实例: {instance_id} | 引擎: {db_type} "
            f"| 监控数据: {len(monitor_results)} 条 | 诊断数据: {len(metric_results)} 条 | 手册: {len(doc_results)} 篇"
        )

        # 拦截决策: 如果两路探针都没捞到线索, 拒绝盲目猜测
        if not metric_results and not monitor_results:
            logger.warning(f"[{trace_id}] 诊断终止: 控制平面未采集到任何有效的监控或诊断数据。")
            yield {
                "type": PacketType.WARNING,
                "content": "⚠️ 故障诊断控制面提示: 由于前置探针未采集到任何监控时序数据或库表诊断数据, 诊断大脑拒绝进行盲目猜测。请检查 Prometheus 连接和数据库连接是否正常。"
            }
            return

        # 构建融合证据链
        knowledge_context = "\n".join(
            [f"- 《{d.get('file_name', '未命名文档')}》: {d.get('text_content', '')}" for d in doc_results]
        ) if doc_results else "当前无匹配的专家 SOP 手册, 请依赖通用运维指标经验进行分析。"

        monitor_context = json.dumps(monitor_results, ensure_ascii=False, indent=2) if monitor_results else "（无 Prometheus 监控数据）"

        system_prompt = await default_prompt.generate(
            get_prompt_config().ops_diagnosis,
            environment=environment,
            db_type=db_type,
            version_code=context.get("version_code", 0),
            db_role=context.get("db_role", "primary"),
            variables=json.dumps(
                {k: v for k, v in context.get("variables", {}).items() if not k.startswith("_")},
                ensure_ascii=False,
            ),
            metric_results=json.dumps(metric_results, ensure_ascii=False, indent=2),
            monitor_results=monitor_context,
            os_log_snapshots=json.dumps(context.get("os_log_snapshots", []), ensure_ascii=False, indent=2),
            knowledge_context=knowledge_context,
            standalone_query=query_text
        )

        yield {"type": PacketType.THOUGHT, "content": "正在召集 DBA 专家大脑融合多路数据, 进行内核级 RCA 根因推演...\n"}

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
                        elif "<" in output_buffer and any(tag.startswith(output_buffer.rsplit("<", 1)[1]) for tag in ["thought>", "/thought>"]):
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
                        elif "<" in output_buffer and "/thought>".startswith(output_buffer.rsplit("<", 1)[1]):
                            break
                        else:
                            yield {"type": PacketType.THOUGHT, "content": output_buffer}
                            output_buffer = ""

            # 处理末尾残留
            if output_buffer.strip():
                m_type = PacketType.THOUGHT if is_thinking else PacketType.ANSWER
                yield {"type": m_type, "content": output_buffer}

        except Exception as e:
            logger.error(f"DBAnalysisSkill 运行异常: {e}")
            yield {"type": PacketType.ERROR, "content": f"⚠️ 分析中断: {str(e)}\n"}
