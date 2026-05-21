import uuid
from loguru import logger
from typing import Any, AsyncGenerator

from skills import BaseSkill
from core.dictionary import PacketType
from agent.common import ContextMemory


class AskGraphSkill(BaseSkill):
    """
    知识图谱检索技能：基于拓扑关系与一二度关联建立的高级结构化检索自治组件。
    完全遵循分布式自治包、小写连字符命名及数据流回填总线规范。
    """
    def __init__(self):
        super().__init__()
        # 默认安全级别，可被 Runtime 覆盖
        self.security_level = 9
        # 延迟导入，防止 NexusCube 核心组件循环依赖
        from services.search.graph_search import GraphBaseSearch
        self.graph_search_service = GraphBaseSearch()

    async def run_stream(
        self,
        context: ContextMemory,
        **kwargs
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        执行图谱拓扑检索任务（完全对接变量注册中心与流式总线版本）
        """
        # 1. 安全提取当前步骤的执行快照与控制信息
        current_execution = context.get("current_execution") or {}
        runtime_skill_name = current_execution.get("skill", "ask-graph-skill")
        output_var = current_execution.get("output_var") or "graph_results"

        current_user = context.get("user_id", "default_user")
        current_agent = context.get("agent_id")
        current_session = context.get("session_id") or uuid.uuid4().hex
        security_level = context.get("security_level", self.security_level)
        
        # 2. 严格从决策控制平面的纯净参数字典（resolved_params）中提取入参，实现与 skill.md 的完全对齐
        resolved_params = current_execution.get("resolved_params") or {}
        
        # 提取或退化降级获取实体词 (vertex_names)
        vertex_names: list[str] = (
            resolved_params.get("vertex_names")
            or context.get("vertex_names") 
            or context.get("entities")
            or [k.strip() for k in context.get("search_keywords", "").split(",") if k.strip()]
        )
        
        # 兜底策略：如果 Planner 或上层未提取出任何实体，则将干净的输入整串作为实体处理
        if not vertex_names:
            query_text = (
                current_execution.get("resolved_input") 
                or getattr(context, 'current_task', None) 
                or context.get("standalone_query") 
                or context.get("question")
            )
            if query_text:
                vertex_names = [query_text]

        # 提取其余业务参数，若缺失则退化到系统默认值
        kb_id = resolved_params.get("kb_id") or context.get("kb_id") or current_execution.get("kb_id")
        search_top_k = resolved_params.get("search_top_k") or context.get("search_top_k", 10)
        max_depth = resolved_params.get("max_depth") or context.get("max_depth", 2)
        
        # 💥 参数对齐映射：将 skill.md 定义的 graph_weight 映射为底层服务所需的 weight
        graph_weight = resolved_params.get("graph_weight") or context.get("graph_weight", 1.2)

        # 3. 边界防御断言
        if not vertex_names:
            yield {"type": PacketType.ERROR, "content": f"{runtime_skill_name}: 变量解析异常，未能在上下文中捕获到任何有效实体词"}
            return

        if not current_agent:
            yield {"type": PacketType.ERROR, "content": f"{runtime_skill_name}: 全局上下文缺失核心身份认证信息 agent_id"}
            return

        if not kb_id:
            yield {"type": PacketType.ERROR, "content": f"{runtime_skill_name}: 缺少核心知识库定位标识 kb_id"}
            return

        # 发送思考状态：开始探索
        entities_str = ", ".join(f"'{v}'" for v in vertex_names)
        yield {"type": PacketType.THOUGHT, "content": f"正在向图谱空间发起拓扑游走，核心实体：[{entities_str}]，最大深度：{max_depth}...\n"}

        try:
            # 4. 调用底层统一图检索服务
            graph_raw_bucket = await self.graph_search_service.search_by_graph(
                kb_id=int(kb_id),
                vertex_names=vertex_names,
                search_top_k=search_top_k,
                weight=graph_weight,
                security=security_level,
                max_depth=max_depth,
                do_rerank=True
            )
            
            # 提取归一化后的 TxtBaseSearchResult 对象
            enriched_refs = graph_raw_bucket.get("rerank_result") or graph_raw_bucket.get("norerank_result") or []

            yield {"type": PacketType.THOUGHT, "content": f"图拓扑游走完毕。已顺沿关系链回表激活 {len(enriched_refs)} 个规范化文本切片...\n"}

            # 5. 格式化并清洗结果
            records_dict = self._build_records(enriched_references=enriched_refs)
            results_list = records_dict["graph_results"]
            
            logger.debug(f"[{runtime_skill_name}] 图谱关联文本记录数: {len(results_list)}")
            
            # 6. 【核心修复】多维数据沉淀与回填总线
            # A. 沉淀至快捷入口（针对 ContextMemory 结构设计）
            if "graph_results" not in context:
                context["graph_results"] = []
            context["graph_results"] = results_list
            
            # B. 注册进变量中心，以便下游 ReasoningSkill 通过模板引擎（如 {{graph_results}}）动态加载
            if "variables" not in context:
                context["variables"] = {}
            context["variables"][output_var] = results_list

            # C. 流式吐出结果包供前端渲染或编排层追踪
            yield {"type": PacketType.GRAPH_RESULTS, "content": results_list}
            
            # D. 交付 DONE 包给 Runtime 总线，宣告单步生命周期完美结束
            yield {"type": PacketType.DONE, "content": results_list}

        except Exception as e:
            logger.error(f"自治图组件 [{runtime_skill_name}] 运行时遭遇严重阻碍: {e}", exc_info=True)
            yield {"type": PacketType.ERROR, "content": f"⚠️ 知识图谱深度检索出现系统级故障: {str(e)}"}

    def _build_records(self, enriched_references: list[Any]) -> dict[str, Any]:
        """
        对齐 TxtBaseSearchResult 规范进行输出，确保下游和标准文本完全等价。
        """
        records = []
        for ref in enriched_references:
            is_dict = isinstance(ref, dict)
            
            content = ref.get('content', '') if is_dict else getattr(ref, 'content', '')
            file_name = (ref.get('title') if is_dict else getattr(ref, 'title', None)) or "Graph Linked File"
            chunk_type = ref.get('chunk_type', 'text') if is_dict else getattr(ref, 'chunk_type', 'text')
            chunk_num = ref.get('chunk_num', 0) if is_dict else getattr(ref, 'chunk_num', 0)
            score = ref.get('score', 0.0) if is_dict else getattr(ref, 'score', 0.0)
            search_helper = ref.get('search_helper', '') if is_dict else getattr(ref, 'search_helper', '')

            meta = (ref.get('metadata') if is_dict else getattr(ref, 'metadata', {})) or {}
            
            record = {
                "title": file_name,
                "content": content,
                "chunk_type": chunk_type,
                "chunk_num": chunk_num,
                "score": score,
                "search_helper": search_helper,
                "metadata": {
                    "chunk_id": meta.get("chunk_id") or (ref.get('chunk_id') if is_dict else getattr(ref, 'chunk_id', '')), 
                    "file_id": meta.get("file_id") or (ref.get('file_id') if is_dict else getattr(ref, 'file_id', '')),
                    "kb_id": meta.get("kb_id") or (ref.get('kb_id') if is_dict else getattr(ref, 'kb_id', '')),
                    "header": meta.get("header") or (ref.get('header', '') if is_dict else getattr(ref, 'header', '')),
                    "page_num": int(meta.get("page_num", 0)),
                    "bbox": meta.get("bbox") or [],
                    "image_name": meta.get("image_name", "")
                }
            }
            records.append(record)
            
        records.sort(key=lambda x: x["score"], reverse=True)
        return {"graph_results": records}