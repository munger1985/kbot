import uuid
import asyncio
from loguru import logger
from typing import Any, AsyncGenerator

from skills import BaseSkill
from agent.memory import MemoryService
from core.dictionary import PacketType
from agent.common import ContextMemory


class AskGraphSkill(BaseSkill):
    """
    知识图谱检索技能：基于拓扑关系与一二度关联建立的高级结构化检索自治组件。
    完全遵循分布式自治包、小写连字符命名及数据流回填总线规范。
    """
    def __init__(self):
        super().__init__()
        self.security_level = 9
        # 延迟导入
        from services.search.graph_search import GraphBaseSearch
        self.graph_search_service = GraphBaseSearch()
        self.memory_service = MemoryService()

    async def run_stream(
        self,
        context: ContextMemory,
        **kwargs
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        执行图谱拓扑检索任务（流式总线版本）
        """
        # 1. 提取当前执行快照（规避硬编码可能带来的系统级 Key 崩溃）
        current_execution = context.get("current_execution") or {}
        runtime_skill_name = current_execution.get("skill", "ask-graph-skill")

        current_user = context.get("user_id", "default_user")
        current_agent = context.get("agent_id")
        current_session = context.get("session_id") or uuid.uuid4().hex
        
        # 2. 智能化获取图谱检索所必需的核心实体词（Vertices）
        # 优先提取上层 NLP 组件/LLM 已经抽好的实体，次之退化到关键词或干净查询词
        vertex_names: list[str] = (
            context.get("vertex_names") 
            or context.get("entities")
            or [k.strip() for k in context.get("search_keywords", "").split(",") if k.strip()]
        )
        
        # 如果上层没有任何实体留下来，把当前的 query_text 整体作为一个实体尝试检索
        if not vertex_names:
            query_text = (
                current_execution.get("resolved_input") 
                or getattr(context, 'current_task', None) 
                or context.get("standalone_query") 
                or context.get("question")
            )
            if query_text:
                vertex_names = [query_text]

        kb_id = context.get("kb_id") or current_execution.get("kb_id")
        search_top_k = context.get("search_top_k", 10)
        weight = context.get("graph_weight", 1.2)  # 图检索默认权重
        max_depth = context.get("max_depth", 2)    # 默认游走2度关系

        # 3. 核心边界防御检查
        if not vertex_names:
            yield {"type": PacketType.ERROR, "content": f"{runtime_skill_name}: 未能在上下文中提取到任何有效的实体或关键词，无法驱动图游走"}
            return

        if not current_agent:
            yield {"type": PacketType.ERROR, "content": f"{runtime_skill_name}: 全局上下文缺失核心身份认证信息 agent_id"}
            return

        if not kb_id:
            yield {"type": PacketType.ERROR, "content": f"{runtime_skill_name}: 缺少核心知识库定位标识 kb_id"}
            return

        # 推送思考状态：开始图拓扑探索
        entities_str = ", ".join(f"'{v}'" for v in vertex_names)
        yield {"type": PacketType.THOUGHT, "content": f"正在向 Oracle 26ai 图谱空间发起拓扑游走，核心实体：[{entities_str}]...\n"}

        try:
            # 4. 调用前面完成的图谱统一检索服务
            # 内部执行纯原生 SQL、CONNECT BY 探索及滑动窗口增强
            graph_raw_bucket = await self.graph_search_service.search_by_graph(
                kb_id=int(kb_id),
                vertex_names=vertex_names,
                search_top_k=search_top_k,
                weight=weight,
                security=self.security_level,
                max_depth=max_depth,
                do_rerank=True  # 默认将其归类进后续的重排池中
            )
            
            # 拿到归一化后的 TxtBaseSearchResult 对象列表
            enriched_refs = graph_raw_bucket.get("rerank_result") or graph_raw_bucket.get("norerank_result") or []

            # 推送思考状态：图谱检索与反查回表完成
            yield {"type": PacketType.THOUGHT, "content": f"图拓扑游走完毕。已通过关系链溯源激活 {len(enriched_refs)} 个关联文本切片...\n"}

            # 5. 格式化并清洗结果，完全对齐公共数据流标准
            records_dict = self._build_records(enriched_references=enriched_refs)
            logger.debug(f"[{runtime_skill_name}] 图谱关联反查出的优质文档记录数: {len(records_dict['graph_results'])}")
            
            # A. 推送前端图谱专用包（如有需要可以渲染成特定前端组件，或供编排层追踪）
            yield {"type": PacketType.GRAPH_RESULTS, "content": records_dict["graph_results"]}
            
            # B. 【核心回填总线】：吐入 DONE 包，供下游 ReasoningSkill 完美无感知吸收
            yield {"type": PacketType.DONE, "content": records_dict["graph_results"]}

        except Exception as e:
            logger.error(f"自治图组件 [{runtime_skill_name}] 运行时遭遇严重阻碍: {e}", exc_info=True)
            yield {"type": PacketType.ERROR, "content": f"⚠️ 知识图谱深度检索出现系统级故障: {str(e)}"}

    def _build_records(self, enriched_references: list[Any]) -> dict[str, Any]:
        """
        对齐 TxtBaseSearchResult 规范进行输出，确保下游和标准文本完全等价。
        """
        records = []
        for ref in enriched_references:
            # 兼容处理：支持对象属性形式或字典形式取值
            content = getattr(ref, 'content', '') or ref.get('content', '') if isinstance(ref, dict) else getattr(ref, 'content', '')
            file_name = getattr(ref, 'title', None) or (ref.get('title') if isinstance(ref, dict) else None) or "Graph Linked File"
            chunk_type = getattr(ref, 'chunk_type', 'text')
            chunk_num = getattr(ref, 'chunk_num', 0)
            score = getattr(ref, 'score', 0.0)
            
            # 获取在 search_by_graph 中被我们魔改注入了关系链路的 search_helper
            search_helper = getattr(ref, 'search_helper', '')

            # 组装元数据
            meta = getattr(ref, 'metadata', {}) or {}
            record = {
                "title": file_name,
                "content": content,
                "chunk_type": chunk_type,
                "chunk_num": chunk_num,
                "score": score,
                "search_helper": search_helper,  # 下游大模型将在这里直接看到图谱关系串
                "metadata": {
                    "chunk_id": getattr(ref, 'chunk_id', ''), 
                    "file_id": getattr(ref, 'file_id', ''),
                    "kb_id": getattr(ref, 'kb_id', ''),
                    "header": getattr(ref, 'header', ''),
                    "page_num": getattr(ref, 'page_num', 0),
                    "bbox": getattr(ref, 'bbox', []),
                    "image_name": getattr(ref, 'image_name', '')
                }
            }
            records.append(record)
            
        # 降序排列
        records.sort(key=lambda x: x["score"], reverse=True)
        return {"graph_results": records}