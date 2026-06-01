# skills/search/ask_graph_skill.py
import uuid
from loguru import logger
from typing import Any, AsyncGenerator

from skills import BaseSkill
from agent.agent.graph_agent import GraphAgent
from core.dictionary import PacketType
from agent.common import ContextMemory

class AskGraphSkill(BaseSkill):
    """
    图谱检索技能组件：全面对齐分布式自治包规范。
    通过 agent_id 懒加载驱动底层的多图谱并行空间游走。
    """
    def __init__(self):
        super().__init__()
        self.security_level = 9
        self.graph_agent = GraphAgent()

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
        tags = context.get("tags") or []
        
        # 2. 从控制平面参数池中提取输入的实体词 (优先拿路由器和规划层抽出来的实体词)
        resolved_params = current_execution.get("resolved_params") or {}
        vertex_names: list[str] = (
            resolved_params.get("vertex_names")
            or context.get("variables", {}).get("extracted_entities")
            or []
        )
        
        # 3. 兜底参数保护：如果模型规划漏掉了实体提取，将整句话作为原始游走入口
        if not vertex_names:
            query_text = (
                current_execution.get("resolved_input") 
                or context.get("standalone_query") 
                or context.get("question")
            )
            if query_text:
                vertex_names = [query_text]

        # 4. 核心边界防御断言
        if not vertex_names:
            content = f"{runtime_skill_name}: 变量解析异常，无法捕获任何有效的实体词网络节点。\n"
            yield {"type": PacketType.ERROR, "content": content}
            return

        if not current_agent:
            content = f"{runtime_skill_name}: 全局记忆体缺失关键参数 agent_id，拒绝下沉查询。\n"
            yield {"type": PacketType.ERROR, "content": content}
            return

        # 推送思考状态包：开始游走
        entities_str = ", ".join(f"'{v}'" for v in vertex_names)
        content = f"正在检索知识图谱空间，深度追踪核心实体：[{entities_str}]...\n"
        yield {"type": PacketType.THOUGHT, "content": content}

        try:
            # 5. 调用核心 Agent 层执行图谱检索（彻底消除 Skill 一层的 kb_id 硬编码）
            enriched_graph_edges = await self.graph_agent.graph_retrieval(
                session_id=current_session,
                agent_id=current_agent,
                question=context.get("question", ""),
                standalone_query=context.get("standalone_query", ""),
                vertex_names=vertex_names,
                security_level=self.security_level,
                user_id=current_user,
                tags=tags
            )

            # 推送思考状态：图游走完成
            content = f"图谱空间游走结束，共召回 {len(enriched_graph_edges)} 组高价值显式实体拓扑路径，启动图结构去重裁剪...\n"
            yield {"type": PacketType.THOUGHT, "content": content}

            # 6. 对齐结果字典组织
            # 按边的关联置信度或者相似度进行降序，保证下游 Reasoning 层最先捕获核心主线
            enriched_graph_edges.sort(key=lambda x: x.get("score", 0.0), reverse=True)

            # 7. 统一向前端下发标准化图包信号，用作图谱组件卡片的高级渲染
            yield {"type": PacketType.GRAPH_RESULTS, "content": enriched_graph_edges}
            logger.debug(f"[{runtime_skill_name}] 图谱分析结果包已成功投递至实时分发队列。")

            # 8. 交付至 Runtime 隔离总线进行记忆体落盘沉淀，供后续生命周期的 Reasoning 总结时使用
            context["graph_results"] = enriched_graph_edges

        except Exception as e:
            logger.error(f"自治组件 [{runtime_skill_name}] 在执行图谱检索生命周期内遭遇致命崩溃: {e}", exc_info=True)
            content = f"⚠️ 图谱网络空间探索发生内部系统故障: {str(e)}\n"
            yield {"type": PacketType.ERROR, "content": content}