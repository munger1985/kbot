import os
import asyncio
import datetime
import json
from typing import Any
from fastapi import Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from dao.entities.kbot_biz_chat_session import KbotBizChatSession
from dao.repositories.kbot_md_chat_session_repo import KbotMdChatSessionRepository
from dao.entities.kbot_biz_chat_session import QAData, Reference
from dao.repositories.kbot_md_agent_repo import KbotMdAgentRepository
from dao.repositories.kbot_md_agent_conf_repo import KbotMdAgentConfRepository
from dao.repositories.kbot_md_prompt_repo import KbotMdPromptRepository
from dao.repositories.kbot_md_chat_history_repo import KbotMdChatHistoryRepository
from dao.entities.kbot_md_chat_history import KbotMdChatHistory
from services.chat.agent_chat import Agent
from services.chat.mcp_chat import Agent as MCPAgent
from loguru import logger
from utils.call_models import CallModel
from api.schemas.agent_schema import AgentChatForm, AgentChatFeedbackForm
from api.schemas.base_response import *


class AgentController:
    
    def _build_references(self, kb_results: list) -> list[Reference]:
        """构建参考文献列表"""
        references = []
        host = os.getenv("KBOT_IP", "localhost")
        port = os.getenv("KBOT_PORT", "8000")
        url = f"http://{host}:{port}"
        
        if kb_results:
            for kb_result in kb_results:
                reference = Reference(
                    chunk_type=kb_result.chunk_type,
                    chunk_file_path=kb_result.chunk_file_path or "",
                    page_num=kb_result.page_num,
                    content=kb_result.content,
                    download_link=f"{url}/api/kb/download?file_id={kb_result.file_id}",
                    preview_link=f"{url}/api/kb/preview?file_id={kb_result.file_id}&page_num={kb_result.page_num}",
                    similarity_score=kb_result.similarity,
                    reranker_score=kb_result.reranker_score
                )
                references.append(reference)
        
        return references
    
    def _build_qa_data(self, question: str, references: list[Reference], by: str, 
                      request_time: str = None) -> QAData:
        """构建 QAData 对象"""
        if request_time is None:
            request_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        
        default_embedding = [0.0] * 2  # Oracle 向量默认值
        
        return QAData(
            question=question,
            answer="",
            qa_embedding=default_embedding,
            references=references,
            feedback=0,
            by=by,
            request_time=request_time,
            response_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        )
    
    def _create_agent(self, agent_id: int, deep_mind: int, security: int = None, tags: list = None):
        """创建 Agent 实例"""
        if deep_mind == 0:
            return Agent(agent_id=agent_id, security=security, tags=tags)
        elif deep_mind == 1:
            return MCPAgent(agent_id=agent_id, security=security, tags=tags)
        else:
            raise ValueError(f"不支持的深度思考版本: {deep_mind}")
    
    async def _get_kb_results(self, agent, question: str, deep_mind: int) -> list:
        """获取知识库结果"""
        if deep_mind == 0:
            return await agent.chat(question=question)
        elif deep_mind == 1:
            tool_results = await agent.chat(question=question)
            kb_results = []
            for tool_result in tool_results:
                if tool_result.kb_results:
                    kb_results.extend(tool_result.kb_results)
            return kb_results
    
    async def _process_session_data(self, session_id: str, agent_id: int, qa_data: QAData) -> bool:
        """处理会话数据写入"""
        sess_repo = KbotMdChatSessionRepository()
        
        # 检查是否已存在会话
        existing_session = await sess_repo.get_session(session_id=session_id)
        
        if existing_session:
            # 追加问答数据
            result = await sess_repo.add_qa_data(session_id=session_id, qa_data=qa_data)
        else:
            # 创建新会话
            session_data = KbotBizChatSession(
                session_id=session_id, 
                agent_id=agent_id, 
                qa_data=[qa_data]
            )
            result = await sess_repo.create_session(session_data)
        
        if result:
            logger.debug(f"成功写入会话数据，session id: {session_id}")
        else:
            logger.warning(f"写入会话数据失败，session id: {session_id}")
        
        return result
    
    async def _get_agent_config(self, agent_id: int) -> tuple:
        """获取智能体配置信息"""
        agent_repo = KbotMdAgentRepository()
        agent = await agent_repo.get_by_id(agent_id)
        
        if agent is None:
            raise ValueError("智能体不存在")
        
        if agent.llm_id is None:
            raise ValueError("智能体配置的 LLM 模型不存在")
        
        # 获取提示词
        prompt_repo = KbotMdPromptRepository()
        prompt_content = await prompt_repo.get_prompt_by_id(agent.prompt_id)
        prompt_template = prompt_content or "根据参考内容回答问题。\n\n参考内容:{context}\n\n回答的问题:{question}"
        
        # 获取模型参数
        model_params = agent.llm_params or {}
        
        return agent, prompt_template, model_params
    
    def _build_context_from_references(self, references: list) -> str:
        """从参考文献构建上下文"""
        context_parts = []
        for ref in references:
            if isinstance(ref, dict):
                content = ref.get('content', '')
            else:
                content = ref.content
            context_parts.append(content)
        
        return "\n".join(context_parts).strip()
    
    def _get_stream_headers(self) -> dict:
        """获取流式响应头"""
        return {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        }
    
    async def _collect_chunks(self, chunk, chunks: list):
        """收集内容块"""
        if isinstance(chunk, str) and chunk.startswith('data: '):
            data_content = chunk[6:].strip()
            if data_content != '[DONE]':
                try:
                    json_data = json.loads(data_content)
                    if "choices" in json_data and len(json_data["choices"]) > 0:
                        if delta := json_data["choices"][0].get("delta"):
                            if content := delta.get("content"):
                                chunks.append(content)
                except json.JSONDecodeError:
                    pass
        elif isinstance(chunk, dict):
            # 处理已经是字典格式的chunk
            if "choices" in chunk and len(chunk["choices"]) > 0:
                if delta := chunk["choices"][0].get("delta"):
                    if content := delta.get("content"):
                        chunks.append(content)
    
    async def _process_final_response(self, chunks, session_id, last_qa_data, agent, agent_repo):
        """处理最终响应并写入数据库"""
        try:
            logger.debug("开始处理最终响应")
            
            # 收集流式输出的所有chunk，并转换为字符串
            str_chunks = []
            for chunk in chunks:
                if isinstance(chunk, bytes):
                    str_chunks.append(chunk.decode("utf-8"))
                else:
                    str_chunks.append(str(chunk))
            
            full_response = "".join(str_chunks) if str_chunks else ""
            logger.debug(f"完整响应长度: {len(full_response)}")

            # 创建聊天历史的数据结构
            app_id = await agent_repo.get_app_id(agent_id=agent.agent_id)
            history = KbotMdChatHistory(
                app_id=app_id,
                agent_id=agent.agent_id,
                session_id=session_id,
                question=last_qa_data["question"],
                answer=full_response,
                created_by=last_qa_data["by"],
                created_time=datetime.datetime.now(),
                updated_by=last_qa_data["by"],
                updated_time=datetime.datetime.now()
            )
            
            # 并行写入Oracle和历史表
            logger.info("开始并行写入数据库")
            results = await asyncio.gather(
                self._write_oracle(session_id, full_response),
                self._write_history(history),
                return_exceptions=True  # 防止一个失败影响另一个
            )
            
            # 检查结果
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"写入任务 {i} 失败: {result}")
                else:
                    logger.debug(f"写入任务 {i} 成功")
                    
            logger.info("成功写入Oracle和聊天历史表")
            
        except Exception as e:
            logger.error(f"处理最终响应时出错: {e}")
    
    async def _write_oracle(self, session_id: str, answer: str) -> None:
        """写入Oracle"""
        try:
            logger.debug(f"正在写入Oracle，session_id: {session_id}")
            sess_repo = KbotMdChatSessionRepository()
            result = await sess_repo.update_last_qa_data_answer(session_id, answer)
            if result:
                logger.debug(f"写入Oracle成功，session_id: {session_id}")
            else:
                logger.warning(f"写入Oracle失败，session_id: {session_id}")
        except Exception as e:
            logger.error(f"写入Oracle错误: {str(e)}")
    
    async def _write_history(self, history: KbotMdChatHistory) -> None:
        """写入历史表"""
        try:
            logger.debug(f"正在写入历史表，session_id: {history.session_id}")
            result = await KbotMdChatHistoryRepository().create(history)
            if result:
                logger.debug(f"写入历史表成功，session_id: {history.session_id}")
            else:
                logger.warning(f"写入历史表失败，session_id: {history.session_id}")
        except Exception as e:
            logger.error(f"记录聊天历史错误: {str(e)}")

    async def agent_chat(self, form: AgentChatForm) -> dict[str, Any]:
        try:
            deep_mind = form.deep_mind or 0
            
            # 创建 Agent
            agent = self._create_agent(
                agent_id=form.agent_id, 
                deep_mind=deep_mind,
                security=form.security_level,
                tags=form.tags
            )
            
            # 获取知识库结果
            kb_results = await self._get_kb_results(agent, form.question, deep_mind)
            
            # 构建参考文献
            references = self._build_references(kb_results)
            
            # 构建 QAData
            qa_data = self._build_qa_data(
                question=form.question,
                references=references,
                by=form.by,
                request_time=form.request_time
            )
            
            # 处理会话数据
            await self._process_session_data(
                session_id=form.session_id,
                agent_id=form.agent_id,
                qa_data=qa_data
            )
            
            # 返回结果
            return {
                "question": qa_data.question,
                "answer": qa_data.answer,
                "qa_embedding": qa_data.qa_embedding,
                "references": [ref.to_dict() for ref in qa_data.references],
                "feedback": qa_data.feedback,
                "by": qa_data.by,
                "request_time": qa_data.request_time,
                "response_time": qa_data.response_time
            }
        
        except Exception as e:
            logger.error(f"Agent chat failed: {e}")
            raise e

    async def agent_stream_chat(
            self,
            request: Request,
            background_tasks: BackgroundTasks,
            session_id: str
        ) -> StreamingResponse | ErrorResponse:
            """处理流式聊天"""
            # 准备数据
            data = await self._prepare_chat_data(session_id)
            if data is None:
                return ErrorResponse(
                    code=status.HTTP_400_BAD_REQUEST,
                    success=False,
                    message="智能体无响应"
                )
            
            last_qa_data, agent, prompt, model_id, model_params, agent_repo = data
            
            async def generate_stream():
                chunks = []
                try:
                    # 调用模型生成流
                    async for chunk in CallModel().call_llm_model(model_id, prompt, **model_params):
                        # 检查客户端是否断开连接
                        if await request.is_disconnected():
                            logger.info("检测到客户端断开连接，结束流")
                            break
                            
                        yield chunk
                        
                        # 收集内容块
                        await self._collect_chunks(chunk, chunks)
                            
                except Exception as e:
                    logger.error(f"流生成错误: {e}")
                finally:
                    # 无论流如何结束，都执行清理
                    if chunks:
                        logger.info(f"收集到 {len(chunks)} 个内容块，执行后台写入")
                        background_tasks.add_task(
                            self._process_final_response,
                            chunks, session_id, last_qa_data, agent, agent_repo
                        )
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers=self._get_stream_headers()
            )
        
    async def _prepare_chat_data(self, session_id: str):
        """准备聊天数据"""
        # 1. 根据session_id从Oracle中获取问答pair
        sess_repo = KbotMdChatSessionRepository()
        logger.debug(f"正在查询 Oracle，session_id: {session_id}")

        last_qa_data = await sess_repo.get_last_qa_data(session_id)
        logger.debug(f"last_qa_data: {last_qa_data}")

        if last_qa_data is None:
            logger.warning("未找到问答记录")
            return None

        refs = last_qa_data["references"]
        agent_id = last_qa_data["agent_id"]
        logger.debug(f"refs: {refs}")
        logger.debug(f"agent_id: {agent_id}")
        
        # 2. 获取智能体配置
        try:
            agent, prompt_template, model_params = await self._get_agent_config(agent_id)
        except ValueError as e:
            logger.warning(str(e))
            return None
            
        model_params["stream"] = True
        
        # 3. 构建上下文和问题
        context = self._build_context_from_references(refs)
        question = last_qa_data["question"]
        prompt = prompt_template.format(context=context.strip(), question=question) # type: ignore
        
        return last_qa_data, agent, prompt, agent.llm_id, model_params, KbotMdAgentRepository()

    async def agent_feedback(self, form: AgentChatFeedbackForm) -> bool:
        try:
            sess_repo = KbotMdChatSessionRepository()
            session_id = form.session_id
            idx= form.question_index
            feedback = form.feedback
            # 根据session_id 和问题索引，更新Oracle对应的问答pair中的feedback数据
            r = await sess_repo.update_qa_feedback(session_id, idx, feedback)
            if r:
                return True
            else:
                return False       
        except Exception as e:  
            raise e
        
    async def agent_get_session(self, session_id: str) -> dict:
        try:
            sess_repo = KbotMdChatSessionRepository()
            # 根据session_id 和问题索引，更新Oracle对应的问答pair中的feedback数据
            r = await sess_repo.get_session(session_id)
            if r:
                return r
            else:
                return {
                    "session_id": session_id, 
                    "qa_data": []
                    }
    
        except Exception as e:  
            raise e
        

    async def agent_del_session(self, session_id: str) -> bool:
        try:
            sess_repo = KbotMdChatSessionRepository()
            # 根据session_id删除聊天会话
            return await sess_repo.delete_session(session_id)
    
        except Exception as e:  
            raise e
        
    async def del_agent(self, agent_id: int, del_prompt: bool = False) -> bool:
        try:
            if del_prompt:
                # 删除agent关联的prompt
                prompt_id = await KbotMdAgentRepository().get_prompt(agent_id)
                if prompt_id:
                    await KbotMdPromptRepository().delete(prompt_id)
                    logger.debug(f"提示词 {prompt_id} 已删除")
                else:
                    logger.debug(f"未找到 智能体 {agent_id} 关联的提示词")
            # 1. 删除agent
            await KbotMdAgentRepository().delete(agent_id)
            logger.debug(f"智能体 {agent_id} 已删除")
            # 2. 删除agent和kb的关联信息
            await KbotMdAgentConfRepository().delete_by_agent_id(agent_id)
            logger.debug(f"智能体 {agent_id} 和知识库的关联信息已删除")
            # 3. 删除agent的聊天会话
            await KbotMdChatSessionRepository().delete_by_agent_id(agent_id)
            logger.debug(f"智能体 {agent_id} 的聊天会话已删除")
            # 4. 删除agent的聊天历史
            await KbotMdChatHistoryRepository().delete_by_agent_id(agent_id)
            logger.debug(f"智能体 {agent_id} 的聊天历史已删除")
            return True

        except Exception as e:
            logger.error(f"智能体 {agent_id} 删除失败: {str(e)}")
            return False
        
    async def agent_chat_dify(self, 
                              agent_id: int, 
                              question: str, 
                              session_id: str,
                              topk: int | None = None,
                              score_threshold: float | None = None,
                        ) -> dict:
        """
        智能体对话 (Dify版)
        # Dify 调用使用深度思考版本的逻辑
        """
        try:
            request_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            logger.info(f"[{request_time}] 开始处理 Dify 请求: {question}")
            
            # 查询向量库获取相关文档
            agent = MCPAgent(agent_id=agent_id, security=9, tags=[]) # security_level 9 表示不校验安全等级
            results  = await agent.chat(question=question, topk=topk, score_threshold=score_threshold)
            
            # 处理知识库结果
            kb_results = []
            for result in results:
                if result.kb_results:
                    kb_results.extend(result.kb_results)
        
            # 构建参考文献列表和 Dify 记录
            references = self._build_references(kb_results)
            records = self._build_dify_records(kb_results)
            
            # 构建 QAData 对象
            qa_data = self._build_qa_data(
                question=question,
                references=references,
                by="dify",
                request_time=request_time
            )

            # 将完整session数据写入Oracle
            await self._process_session_data(session_id, agent_id, qa_data)

            return {"records": records}
            
        except Exception as e:
            logger.error(f"Agent chat dify failed: {e}")
            raise e
    
    def _build_dify_records(self, kb_results: list) -> list[dict]:
        """构建 Dify 返回记录"""
        records = []
        if kb_results:
            for kb_result in kb_results:
                record = {
                    "metadata": {
                        "path": kb_result.chunk_file_path or "",
                        "description": f"page: {kb_result.page_num}"
                    },
                    "score": kb_result.similarity,
                    "title": kb_result.file_id,
                    "content": kb_result.content
                }
                records.append(record)
        return records


# 创建控制器实例
agent_controller = AgentController()