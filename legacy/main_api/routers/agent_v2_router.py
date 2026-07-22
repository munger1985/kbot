"""Explicit V2 knowledge retrieval route; no V1 fallback."""
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent.agent.document_agent_v2 import DocumentAgentV2
from agent.agent.root_agent_v2 import RootAgentV2
from api.schemas.base_response import SuccessResponse
from platform_core.auth.shortcuts import AnyAuth
from platform_core.config.settings import get_knowledge_core_config
from knowledge_core.application.task_dto import KnowledgeTask
from knowledge_core.application.answer_generation import LLMAnswerGenerator
from skills.knowledge_retrieval_v2 import KnowledgeRetrievalSkillV2
from services.basic.agent_service import AgentService

router = APIRouter(prefix="/agent/v2", tags=["Agent Knowledge V2"])
_kc_config = get_knowledge_core_config()
_agent_service = AgentService()


async def _resolve_answer_model(task: KnowledgeTask) -> str:
    params = await _agent_service.get_agent_model_params(task.agent_id)
    return params.llm_model


_root_agent_v2 = RootAgentV2(document_agent=DocumentAgentV2(
    retrieval_skill=KnowledgeRetrievalSkillV2(kc_url=f"http://{_kc_config.service_host}:{_kc_config.service_port}"),
), answer_generator=LLMAnswerGenerator(model_resolver=_resolve_answer_model))


class KnowledgeChatV2Request(BaseModel):
    by: str = Field(min_length=1, max_length=256)
    agent_id: int
    domain_id: int
    collection_ids: list[int] = Field(min_length=1, max_length=128)
    question: str = Field(min_length=1, max_length=8000)
    standalone_query: str | None = None
    security_level: int = Field(default=3, ge=0, le=3)
    session_id: str | None = None
    parent_run_id: str | None = None


def _task(payload: KnowledgeChatV2Request) -> KnowledgeTask:
    return KnowledgeTask(
        task_id=str(uuid4()), parent_run_id=payload.parent_run_id or str(uuid4()),
        domain_id=payload.domain_id, agent_id=payload.agent_id,
        original_query=payload.question, standalone_query=payload.standalone_query or payload.question,
        collection_ids=tuple(sorted(set(payload.collection_ids))), security_level=payload.security_level,
    )


@router.post("/knowledge/streaming", response_class=StreamingResponse, status_code=status.HTTP_200_OK)
async def knowledge_v2_stream(payload: KnowledgeChatV2Request, auth: AnyAuth, background_tasks: BackgroundTasks):
    return StreamingResponse(_root_agent_v2.stream(_task(payload)), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no",
    })


@router.post("/knowledge", response_model=SuccessResponse, status_code=status.HTTP_200_OK)
async def knowledge_v2_nonstream(payload: KnowledgeChatV2Request, auth: AnyAuth):
    result = await _root_agent_v2.complete(_task(payload))
    return SuccessResponse(data=result, message="Knowledge Core V2 grounded answer completed")
