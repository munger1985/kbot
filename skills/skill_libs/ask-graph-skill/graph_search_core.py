# skills/search/ask_graph_skill.py
import uuid
from loguru import logger
from typing import Any, AsyncGenerator

from skills import BaseSkill
from agent.agent.graph_agent import GraphAgent
from core.dictionary import PacketType
from agent.common import ContextMemory
from services.graph import GraphService

class AskGraphSkill(BaseSkill):
    """
    图谱检索技能组件：全面对齐分布式自治包规范。
    通过 agent_id 懒加载驱动底层的多图谱并行空间游走。
    """
    def __init__(self):
        super().__init__()
        self.graph_agent = GraphAgent()
        self.graph_service = GraphService()

    async def run_stream(
        self,
        context: ContextMemory,
        **kwargs
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        执行图谱检索任务（镜像对齐隔离沙箱版本）
        """
        # 1. 提取当前执行快照（保护核心总线）
        current_execution = context.get("current_execution") or {}
        runtime_skill_name = current_execution.get("skill", "ask-graph-skill")

        current_user = context.get("user_id", "default_user")
        current_agent = context.get("agent_id")
        current_session = context.get("session_id") or uuid.uuid4().hex
        current_security_level = context.get("security_level") or 0
        tags = context.get("tags") or []
        
        # 2. 从控制平面参数池中提取输入的实体词 (优先拿路由器和规划层抽出来的实体词)
        resolved_params = current_execution.get("resolved_params") or {}
        variables = context.get("variables") or {}
        
        # 建立高容错的提取管道
        raw_source: list[str] | str | None = (
            # 1. 编排层动态注入的显式入参
            resolved_params.get("vertex_names")
            or resolved_params.get("keywords")
            
            # 2. 从变量中心 (Variables Registry) 容错反查各种可能的 Key
            or variables.get("extracted_entities")
            or variables.get("search_keywords")
            or variables.get("keywords")
            
            # 3. 直接回溯到全局上下文顶层的 search_keywords 字段
            or context.get("search_keywords")
        )

        # 归一化处理：清洗上面捞出来的各种奇葩数据格式（可能是 list，可能是逗号分隔的 str）
        raw_keywords: list[str] = []
        if raw_source:
            if isinstance(raw_source, list):
                raw_keywords = [str(item).strip() for item in raw_source if item]
            elif isinstance(raw_source, str):
                # 兼容 "轮廓仪法, 半导体晶圆测量" 这种逗号分隔的字符串
                split_char = "," if "," in raw_source else " "
                raw_keywords = [item.strip() for item in raw_source.split(split_char) if item.strip()]

        # 4. 终极参数防御：如果上述漏斗全部踩空，直接借用改写后的 Standalone Query 或者原始 question
        if not raw_keywords:
            fallback_text = (
                context.get("standalone_query") 
                or context.get("question")
            )
            if fallback_text:
                logger.warning(f"[{runtime_skill_name}] 核心实体漏斗全部踩空，触发终极防御：直接将文本整体作为输入")
                raw_keywords = [fallback_text]

        # 5. 核心边界断言
        if not raw_keywords:
            yield {"type": PacketType.ERROR, "content": f"{runtime_skill_name}: 无法从上下文提炼任何检索线索。"}
            return
        if not current_agent:
            yield {"type": PacketType.ERROR, "content": f"{runtime_skill_name}: 缺失全局变量 agent_id。"}
            return

        # 推送思考状态：准备进入向量实体消歧
        yield {
            "type": PacketType.THOUGHT, 
            "content": f"分析执行上下文，捕获到检索线索: {raw_keywords}。正在基于 Oracle 26ai 向量库进行图谱实体消歧对齐...\n"
        }

        try:
            # ========================================================
            # 🎯 核心机制升级：通过关键词向量近义反查图谱里的真实 VERTEX_NAME
            # ========================================================
            aligned_vertex_names = []
            aligned_vertex_names = await self.graph_service.align_vertices_by_embedding(
                keywords=raw_keywords,
                agent_id=current_agent,
                top_k=2
            )
            
            # 如果向量对齐层未就绪或者未匹配到，则降级使用清洗后的 raw_keywords
            if not aligned_vertex_names:
                aligned_vertex_names = raw_keywords

            entities_str = ", ".join(f"'{v}'" for v in aligned_vertex_names)
            yield {
                "type": PacketType.THOUGHT, 
                "content": f"语义实体对齐成功。正在驱动图谱网络进行多维路径游走，目标核心节点: [{entities_str}]...\n"
            }

            # ========================================================
            # 6. 下沉进入原生 SQL/PGQ 原生图拓扑检索
            # ========================================================
            enriched_graph_edges = await self.graph_agent.graph_retrieval(
                session_id=current_session,
                agent_id=current_agent,
                question=context.get("question", ""),
                standalone_query=context.get("standalone_query", ""),
                vertex_names=aligned_vertex_names,  # 🎯 传入洗干净且对齐后的真实图实体
                security_level=current_security_level,
                user_id=current_user,
                tags=tags
            )

            # 后续原有的排序、投递及沉淀逻辑...
            enriched_graph_edges.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            yield {"type": PacketType.GRAPH_RESULTS, "content": enriched_graph_edges}
            
            # 数据沉淀回 ContextMemory 指定的数据结构槽位中
            context["graph_results"] = enriched_graph_edges

        except Exception as e:
            logger.error(f"自治组件 [{runtime_skill_name}] 执行失败: {e}", exc_info=True)
            yield {"type": PacketType.ERROR, "content": f"⚠️ 图谱空间检索中断: {str(e)}\n"}