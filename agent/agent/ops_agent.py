import uuid
from typing import Any
from loguru import logger
from fastapi import BackgroundTasks
from fastapi.responses import StreamingResponse

from core.dictionary import PacketType
from agent.orchestrator import OpsOrchestrator
from agent.common import AgentStreamMixin
from services.basic import AgentService


class OpsAgent(AgentStreamMixin):
    """
    智能故障自愈(AIOps)智能体大脑 - 流式核心网关版 (3.5 纯净平移版)
    - 职责纯粹化：不越级处理任何 RAG、变量提取、模型拼接，将其全面托管给 OpsOrchestrator。
    - 接口拉齐化：全面对齐 RootAgent 规范，提供纯正的流式 text/event-stream 输出。
    """

    def __init__(self):
        # 核心：持有新一代自愈流水线执行面编排器
        self.orchestrator = OpsOrchestrator()
        self.agent_service = AgentService()

    async def chat(
        self,
        background_tasks: BackgroundTasks,
        user_id: str,
        agent_id: int,
        instance_id: str,
        query: str,
        session_id: str | None = None,
        client_time: str | None = None,
        client_tz: str | None = None,
        images_base64: list[str] = [],
    ) -> StreamingResponse:
        """
        AIOps 智能运维统一接口入口（流式 text/event-stream 返回版）
        """
        # 1. 规范并收拢会话 ID
        if not session_id or session_id == "new_session":
            session_id = f"sess_{uuid.uuid4().hex[:12]}"
        
        # 2. 预生成 message_id，用于前端消息去重与生命周期绑定
        message_id = str(uuid.uuid4())

        logger.info(f"[OpsAgent] 接收到自愈请求 | User: {user_id} | Session: {session_id} | MsgID: {message_id}")

        async def event_generator():
            try:
                # Step 1. 握手阶段：立即发送元数据包（不需要模拟延迟）
                yield self._format_sse(
                    event_type=PacketType.METADATA,
                    content={"session_id": session_id, "message_id": message_id}
                )

                # Step 2. 视觉搜索：检测到图片时，先做双向互检索
                enriched_query = query
                all_images = [img for img in images_base64 if img and len(img) > 100]
                if all_images:
                    try:
                        from services.visual.search_engine import VisualSearchEngine
                        engine = VisualSearchEngine()
                        kb_ids = await self.agent_service.get_kb_list(int(agent_id))
                        all_visual_results = []
                        for idx, img in enumerate(all_images):
                            try:
                                results = await engine.search(
                                    query=query, image_base64=img, top_k=5,
                                    kb_ids=kb_ids,
                                )
                                all_visual_results.extend(results)
                            except Exception as e:
                                logger.warning(f"[VisualSearch] 第 {idx+1} 张图片搜索跳过: {e}")

                        if all_visual_results:
                            seen = set()
                            unique_results = []
                            for r in sorted(all_visual_results, key=lambda x: x.similarity, reverse=True):
                                key = f"{r.file_id}:{r.page_no}"
                                if key not in seen:
                                    seen.add(key)
                                    unique_results.append(r)
                            parts = [f"用户问题: {query}\n\n以下是通过图片搜索找到的相关文档页面的图文内容:"]
                            for i, r in enumerate(unique_results[:10]):
                                parts.append(
                                    f"\n### 页面 {i+1} (相似度: {r.similarity:.2f})\n"
                                    f"图片路径: {r.page_image_path}\n"
                                    f"描述: {r.image_description}\n"
                                    f"文本内容: {' '.join(r.text_snippets[:3])}"
                                )
                            enriched_query = "\n".join(parts)
                            logger.info(f"[VisualSearch] 图片搜索完成, 找到 {len(unique_results)} 个结果")
                    except Exception as e:
                        logger.warning(f"[VisualSearch] 图片搜索跳过: {e}")

                # Step 3. 直接获取下层强类型编排器的通用自愈数据流
                # 内部已闭环：确保会话就绪 -> Planner 生成计划 -> 变量动态替换提取 -> 驱动 SkillRuntime 状态机 -> 异步反思审计
                raw_pipeline = self.orchestrator.execute_ops_stream_pipeline(
                    background_tasks=background_tasks,
                    user_id=user_id,
                    session_id=session_id,
                    agent_id=agent_id,
                    question=enriched_query,
                    trigger_type="manual",
                    instance_id=instance_id,
                    client_time=client_time,
                    client_tz=client_tz,
                )

                # Step 3. 通过"平滑滤镜"将下层抛上来的包，按需转化为给前端的逐字模拟流
                async for packet_type, content in self._smooth_stream_pipeline(raw_pipeline):
                    yield self._format_sse(
                        event_type=packet_type,
                        content=content,
                        message_id=message_id
                    )

            except Exception as e:
                logger.exception(f"[OpsAgent] 顶层自愈流水线遭遇崩溃: {str(e)}")
                # 异常时输出故障警告（走打字机模拟器，给前端极佳的体验）
                fallback_msg = "内核诊断层遇到非预期阻碍，自愈控制面已安全熔断，请联系底层 DBA 专家排查。"
                async for p_type, content in self._simulate_char_stream(PacketType.ERROR, fallback_msg):
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

    async def resume(
        self,
        background_tasks: BackgroundTasks,
        request_id: str,
        user_data: dict[str, Any] | None = None,
        user_note: str | None = None,
        user_error: str | None = None,
    ) -> StreamingResponse:
        """AIOps HITL 恢复执行接口 — 从挂起点恢复诊断流水线"""
        message_id = str(uuid.uuid4())
        logger.info(f"[OpsAgent] HITL 恢复请求 | RequestID: {request_id} | MsgID: {message_id}")

        async def event_generator():
            try:
                yield self._format_sse(
                    event_type=PacketType.METADATA,
                    content={"message_id": message_id, "status": "resuming"}
                )
                raw_pipeline = self.orchestrator.resume_ops_stream_pipeline(
                    background_tasks=background_tasks,
                    request_id=request_id,
                    user_data=user_data,
                    user_note=user_note,
                    user_error=user_error,
                )
                async for packet_type, content in self._smooth_stream_pipeline(raw_pipeline):
                    yield self._format_sse(event_type=packet_type, content=content, message_id=message_id)
            except Exception as e:
                logger.exception(f"[OpsAgent] HITL 恢复流水线崩溃: {str(e)}")
                async for p_type, content in self._simulate_char_stream(
                    PacketType.ERROR, "内核诊断层在恢复过程中遇到非预期阻碍。"
                ):
                    yield self._format_sse(event_type=p_type, content=content, message_id=message_id)

        return StreamingResponse(
            event_generator(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
        )

    async def approve(
        self,
        background_tasks: BackgroundTasks,
        request_id: str,
        approved: bool,
        approver_note: str | None = None,
    ) -> StreamingResponse:
        """AIOps HITL 审批接口 — 用户审批后恢复或终止变更执行"""
        message_id = str(uuid.uuid4())
        logger.info(f"[OpsAgent] HITL 审批 | RequestID: {request_id} | Approved: {approved}")

        async def event_generator():
            try:
                yield self._format_sse(
                    event_type=PacketType.METADATA,
                    content={"message_id": message_id, "status": "approving"}
                )
                raw_pipeline = self.orchestrator.resume_with_approval(
                    background_tasks=background_tasks,
                    request_id=request_id,
                    approved=approved,
                    approver_note=approver_note,
                )
                async for packet_type, content in self._smooth_stream_pipeline(raw_pipeline):
                    yield self._format_sse(event_type=packet_type, content=content, message_id=message_id)
            except Exception as e:
                logger.exception(f"[OpsAgent] 审批恢复流水线崩溃: {str(e)}")
                async for p_type, content in self._simulate_char_stream(
                    PacketType.ERROR, "审批处理过程中遇到非预期错误。"
                ):
                    yield self._format_sse(event_type=p_type, content=content, message_id=message_id)

        return StreamingResponse(
            event_generator(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
        )
