import uuid
from loguru import logger
from typing import Any, AsyncGenerator
from fastapi import BackgroundTasks

from skills import BaseSkill
from agent.memory import MemoryService
from core.dictionary import PacketType
from agent.common import ContextMemory


class AskDocSkill(BaseSkill):
    """
    文档检索技能：完全遵循分布式自治包、小写连字符命名及数据流回填总线规范。
    """
    def __init__(self):
        super().__init__()
        self.security_level = 9
        # 延迟导入避免循环依赖
        from agent.agent import DocAgent
        self.doc_agent = DocAgent()
        self.memory_service = MemoryService()

    async def run_stream(
        self,
        context: ContextMemory,
        **kwargs
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        执行文档检索任务（高鲁棒性流式总线版本）
        """
        # 1. 提取当前执行快照快照（规避硬编码硬取可能带来的系统级 Key 崩溃）
        current_execution = context.get("current_execution") or {}
        runtime_skill_name = current_execution.get("skill", "ask-doc-skill")

        current_user = context.get("user_id", "default_user")
        current_agent = context.get("agent_id")
        current_session = context.get("session_id") or uuid.uuid4().hex
        
        # 2. 智能化、高置信度获取被基座 Runtime 替换好变量后的干净查询词
        query_text = (
            current_execution.get("resolved_input") 
            or getattr(context, 'current_task', None) 
            or context.get("standalone_query") 
            or context.get("question")
        )
        
        search_keywords = context.get("search_keywords", "")
        tags = context.get("tags") or []
        
        if not query_text:
            yield {"type": PacketType.ERROR, "content": f"{runtime_skill_name}: 变量解析异常，未能获取到任何有效的检索文本"}
            return

        if not current_agent:
            yield {"type": PacketType.ERROR, "content": f"{runtime_skill_name}: 全局上下文缺失核心身份认证信息 agent_id"}
            return

        # 3. 异步任务句柄优先采用总线透传，次之本地隐式构建
        bg_tasks = kwargs.get("background_tasks") or BackgroundTasks()

        # 推送思考状态：开始检索
        yield {"type": PacketType.THOUGHT, "content": f"正在检索知识库文档，查询词：'{query_text}'...\n"}

        try:
            # 4. 执行底层 RAG 检索
            enriched_refs = await self.doc_agent.rag_retrieval(
                background_tasks=bg_tasks,
                session_id=current_session,
                agent_id=current_agent,
                question=context.get("question", query_text), # 传入原始问题或当前文本供参考
                standalone_query=query_text,
                search_keywords=search_keywords,
                security_level=self.security_level,
                user_id=current_user,
                tags=tags
            )
            
            # 推送思考状态：检索完成，正在重排
            yield {"type": PacketType.THOUGHT, "content": f"已在知识空间寻获 {len(enriched_refs)} 个关联文本碎片，正在启动混合重排...\n"}

            # 5. 格式化并清洗结果
            records_dict = self._build_records(enriched_references=enriched_refs)
            logger.debug(f"[{runtime_skill_name}] 格式化后的优质文档记录数: {len(records_dict['doc_results'])}")
            
            # A. 推送最终前端渲染或编排层追踪需要的结果包
            yield {"type": PacketType.DOC_RESULTS, "content": records_dict["doc_results"]}
            
            # B. 【核心回填设计】：将最干净的数据对象塞入 DONE 包，交付给 Runtime 总线。
            # 这能保障执行链条的下一棒（ReasoningSkill 或者是 ChartRender）可以直接在变量池中享用这份记录。
            yield {"type": PacketType.DONE, "content": records_dict["doc_results"]}

        except Exception as e:
            logger.error(f"自治组件 [{runtime_skill_name}] 运行时遭遇严重阻碍: {e}", exc_info=True)
            yield {"type": PacketType.ERROR, "content": f"⚠️ 知识库检索出现系统级故障: {str(e)}"}

    def _build_records(self, enriched_references: list[dict]) -> dict[str, Any]:
        """
        根据 TxtBaseSearchResult 的定义格式化输出。
        确保下游 ReasoningSkill 能够获得完整的元数据用于溯源。
        """
        records = []
        for ref in enriched_references:
            content = ref.get("content", "")
            
            record = {
                # 基础展示信息
                "title": ref.get("file_name", "Unknown File"),
                "content": content,
                "chunk_type": ref.get("chunk_type", "text"),
                "chunk_num": ref.get("chunk_num", 0),

                # 评分体系（优先采用交叉高维重排分，次之采用原始向量距离分）
                "score": ref.get("rerank_score") if ref.get("rerank_score", 0) > 0 else ref.get("score", 0),
                
                # 扩展元数据 (供前端渲染高亮或点击跳转定位)
                "metadata": {
                    "chunk_id": ref.get("chunk_id") or ref.get("id"), 
                    "file_id": ref.get("file_id"),
                    "kb_id": ref.get("kb_id"),
                    "header": ref.get("header", ""),
                    "page_num": ref.get("page_num", 0),
                    "bbox": ref.get("bbox", []),
                    "image_name": ref.get("image_name", "")
                }
            }
            records.append(record)
            
        # 按分数从高到低排序，确保推理层先看到最相关的片段
        records.sort(key=lambda x: x["score"], reverse=True)
        return {"doc_results": records}