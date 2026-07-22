import uuid
import aiohttp
from loguru import logger
from fastapi import BackgroundTasks
from fastapi.responses import StreamingResponse
from services.basic import AgentService
from api.schemas.agent_schema import *
from api.schemas.base_response import SuccessResponse
from agent.agent import RootAgent, DifyService
from platform_core.exceptions import *
from platform_core.config.settings import get_ask_data_api_config
from services.basic import PromptService


class AgentController:
    def __init__(self):
        self.agent_service = AgentService()
        self.dify_service = DifyService()
        self.root_agent = RootAgent()
        self.prompt_service = PromptService()

    async def feedback(self, form: AgentChatFeedbackForm) -> SuccessResponse:
        """Submits user feedback for a chat record."""
        await self.agent_service.feedback(form.memory_id, form.feedback)
        return SuccessResponse(message="Feedback submitted successfully")
        
    async def get_conversation_context(self, session_id: str) -> SuccessResponse:
        """Retrieves history for a specific session."""
        records = await self.agent_service.get_context_by_session(session_id)
        return SuccessResponse(data=records, message="Conversation context retrieved")

    async def remove_session(self, session_id: str) -> SuccessResponse:
        """Deletes a chat session."""
        await self.agent_service.remove_session(session_id)
        return SuccessResponse(message="Session deleted")
        
    async def remove_agent(self, agent_id: int, del_prompt: bool = False) -> SuccessResponse:
        """Removes an agent and its configurations."""
        await self.agent_service.remove_agent(agent_id, del_prompt)
        return SuccessResponse(message=f"Agent {agent_id} deleted")

    async def agent_chat_nonstream(self, form: AgentChatForm, background_tasks: BackgroundTasks) -> SuccessResponse:
        """
        Agent interaction (Non-streaming).
        Returns the formatted dictionary with answer, embedding, and timestamps.
        """
        stream_response = await self.root_agent.chat(
            background_tasks=background_tasks,
            session_id=form.session_id,
            user_id=form.by,
            agent_id=form.agent_id,
            query=form.question,
            security_level=form.security_level,
            tags=form.tags or [],
            images_base64=form.images_base64,
        )
        
        full_answer = ""
        
        # 遍历异步生成器，拼接完整的大模型回复
        async for chunk in stream_response.body_iterator:
            if isinstance(chunk, bytes):
                full_answer += chunk.decode("utf-8")
            else:
                full_answer += str(chunk)

        return SuccessResponse(data=full_answer, message="Agent chat successful")
    
    async def agent_chat_stream(self, form: AgentChatForm, background_tasks: BackgroundTasks) -> StreamingResponse:
        """
        Agent interaction (Streaming).
        Uses BackgroundTasks to handle database persistence after the stream starts.
        """
        return await self.root_agent.chat(
            background_tasks=background_tasks,
            session_id=form.session_id,
            user_id = form.by,
            agent_id=form.agent_id,
            query=form.question,
            security_level=form.security_level,
            tags=form.tags or [],
            images_base64=form.images_base64,
        )
    
    async def dify_search(self, form: DifySearchForm, background_tasks: BackgroundTasks) -> dict:
        """
        Dify search.
        Uses BackgroundTasks to handle database persistence after the stream starts.
        """
        agent_id = int(form.knowledge_id)
        session_id = form.retrieval_setting.get("session_id") or uuid.uuid4().hex
        security_level = form.retrieval_setting.get("security_level") or 9
        user_id = form.retrieval_setting.get("user_id") or "dify_system"
        
        return await self.dify_service.search(
                    agent_id=agent_id, 
                    question=form.query, 
                    session_id=session_id,
                    security_level=security_level,
                    user_id=user_id
                )

    async def get_conversation_list(self, user_id: str) -> SuccessResponse:
        """Retrieves a list of all chat records associated with a specific `user_id`."""
        convs = await self.agent_service.get_conversation_list(user_id)
        return SuccessResponse(data=convs, message="Conversation list retrieved")
    
    async def rename_conversation(self, form: AgentRenameConversationForm) -> SuccessResponse:
        """Renamesames a chat session title in the database."""
        await self.agent_service.rename_conversation(form.session_id, form.new_title)
        return SuccessResponse(message="Session renamed")

    async def list_profiles(self) -> SuccessResponse:
        """获取 AIReport SelectAI 的 profile 列表。

        通过外部问数 API 的管理端点获取所有可用的 profile 元数据，
        供前端配置 agent 时选择对应的数据源 profile。
        """
        api_config = get_ask_data_api_config()
        api_key = api_config.api_key
        profiles_endpoint = api_config.profiles_endpoint
        timeout = aiohttp.ClientTimeout(total=api_config.timeout)

        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        logger.info(f"[AgentController] 请求 profile 列表 | endpoint={profiles_endpoint}")

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(profiles_endpoint, headers=headers) as response:
                    if response.status != 200:
                        err_text = await response.text()
                        logger.error(f"[AgentController] 获取 profile 列表失败 {response.status}: {err_text}")
                        raise InternalServerError(f"外部接口返回异常状态 {response.status}")

                    res_json = await response.json()
                    return SuccessResponse(data=res_json, message="Profile list retrieved")

        except aiohttp.ClientConnectorError:
            logger.error(f"[AgentController] 无法连接 profile 列表接口: {profiles_endpoint}")
            raise InternalServerError("无法连接至外部 profile 接口")
        except InternalServerError:
            raise
        except Exception as e:
            logger.exception(f"[AgentController] 获取 profile 列表异常: {e}")
            raise InternalServerError(f"获取 profile 列表失败: {str(e)}")

    async def reset_sys_prompt(self) -> SuccessResponse:
        """重置系统提示词"""
        await self.prompt_service.reset_sys_prompt()
        return SuccessResponse(message="System prompt reset successfully")
    
# initialize the controller
agent_controller = AgentController()