import uuid
from loguru import logger
from fastapi import BackgroundTasks
from fastapi.responses import StreamingResponse
from platform_core.dictionary import PacketType
from agent.common import AgentStreamMixin
from services.basic import AgentService


class RootAgent(AgentStreamMixin):
    def __init__(self):
        from agent.orchestrator import RootOrchestrator
        from agent.memory.context_manager import ContextManager
        from agent.memory import MemoryService
        
        self.orchestrator = RootOrchestrator()
        self.context_manager = ContextManager()
        self.memory_service = MemoryService()
        self.agent_service = AgentService()

    async def chat(
        self, 
        background_tasks: BackgroundTasks, 
        user_id: str, 
        agent_id: int, 
        query: str, 
        session_id: str | None = None,
        security_level: int = 1,
        tags: list[str] = [],
        images_base64: list[str] = [],
    ) -> StreamingResponse:
        """
        API 调用的统一入口（流式返回版）
        """
        # 1. 确保 session_id 存在
        if not session_id or session_id == "new_session":
            session_id = f"sess_{uuid.uuid4().hex[:12]}"
        
        # 2. 预生成 message_id，用于前端消息去重与生命周期绑定
        message_id = str(uuid.uuid4())

        # 读取智能体的视觉模型配置
        visual_model = ""
        try:
            model_params = await self.agent_service.get_agent_model_params(agent_id)
            visual_model = model_params.visual_embedding_model
        except Exception as e:
            logger.warning(f"[RootAgent] 读取智能体视觉模型配置失败: {e}")

        logger.info(
            f"[Chat入参] user={user_id} agent={agent_id} session={session_id} "
            f"security_level={security_level} tags={tags} "
            f"images={len(images_base64)} query={query[:80]}"
        )

        async def event_generator():
            try:
                # 1. 握手阶段：立即发送元数据包（不需要模拟延迟）
                yield self._format_sse(
                    event_type=PacketType.METADATA, 
                    content={"session_id": session_id, "message_id": message_id}
                )

                # 2. 确保内存上下文中会话已初始化
                await self.memory_service.ensure_session_exists(
                    session_id=session_id, user_id=user_id, agent_id=agent_id, question=query
                )

                # 2.5 视觉搜索：检测到图片时，先做双向互检索
                enriched_query = query
                all_images = [img for img in images_base64 if img and len(img) > 100]

                if all_images:
                    try:
                        from services.visual.search_engine import VisualSearchEngine
                        engine = VisualSearchEngine()
                        # 获取智能体关联的知识库，限定图片搜索范围
                        kb_ids = await self.agent_service.get_kb_list(agent_id)
                        all_visual_results = []
                        for idx, img in enumerate(all_images):
                            try:
                                results = await engine.search(
                                    query=query, image_base64=img, top_k=3,
                                    visual_model=visual_model,
                                    kb_ids=kb_ids,
                                )
                                all_visual_results.extend(results)
                            except Exception as e:
                                logger.warning(f"[VisualSearch] 第 {idx+1} 张图片搜索跳过: {e}")

                        if all_visual_results:
                            # 去重并按相似度排序
                            seen = set()
                            unique_results = []
                            for r in sorted(all_visual_results, key=lambda x: x.similarity, reverse=True):
                                key = f"{r.file_id}:{r.page_no}"
                                if key not in seen:
                                    seen.add(key)
                                    unique_results.append(r)

                            parts = [f"用户问题: {query}\n\n以下是通过 {len(all_images)} 张图片搜索找到的相关文档页面的图文内容:"]
                            for i, r in enumerate(unique_results[:10]):
                                parts.append(
                                    f"\n### 页面 {i+1} (相似度: {r.similarity:.2f})\n"
                                    f"图片路径: {r.page_image_path}\n"
                                    f"描述: {r.image_description}\n"
                                    f"文本内容: {' '.join(r.text_snippets[:3])}"
                                )
                            enriched_query = "\n".join(parts)
                            logger.info(f"[VisualSearch] {len(all_images)} 张图片搜索完成, 去重后 {len(unique_results)} 个结果, enriched_query_len={len(enriched_query)}")
                            for i, r in enumerate(unique_results[:5]):
                                logger.info(f"[VisualSearch]   结果[{i}]: file={r.file_id} page={r.page_no} sim={r.similarity:.4f} text_snippets={len(r.text_snippets)} source={r.source}")
                    except Exception as e:
                        logger.warning(f"[VisualSearch] 图片搜索跳过: {e}")

                # 3. 获取下层编排器的原始数据流
                raw_pipeline = self.orchestrator.chat_stream_pipeline(
                    background_tasks=background_tasks,
                    user_id=user_id,
                    session_id=session_id,
                    agent_id=agent_id,
                    question=enriched_query,
                    security_level=security_level,
                    tags=tags
                )

                # 4. 通过“平滑滤镜”将下层完整的包，按需转化为给前端的逐字模拟流
                async for packet_type, content in self._smooth_stream_pipeline(raw_pipeline):
                    yield self._format_sse(
                        event_type=packet_type, 
                        content=content, 
                        message_id=message_id
                    )

            except Exception as e:
                logger.exception(f"RootAgent 顶层流式处理失败: {str(e)}")
                # 异常时输出完整的错误信息（走模拟器，给前端极佳的打字机体验）
                async for p_type, content in self._simulate_char_stream(PacketType.ERROR, "系统处理请求时遇到困难，请稍后再试"):
                    yield self._format_sse(event_type=p_type, content=content, message_id=message_id)

        return StreamingResponse(
            event_generator(), 
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
