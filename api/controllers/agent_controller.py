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
    async def agent_chat(self, form: AgentChatForm) -> dict[str, Any]:
        try:
            # 确定 API 版本
            deep_mind = form.deep_mind or 0
            kb_results = []
            if deep_mind == 0:
                agent = Agent(agent_id=form.agent_id, security=form.security_level, tags=form.tags)
                kb_results = await agent.chat(question=form.question)
            elif deep_mind == 1:
                # 处理深度思考版本的逻辑
                agent = MCPAgent(agent_id=form.agent_id, security=form.security_level, tags=form.tags)
                tool_results = await agent.chat(question=form.question)
                # 目前只处理知识库结果
                for tool_result in tool_results:
                    if tool_result.kb_results:
                        kb_results.extend(tool_result.kb_results)
            else:
                raise ValueError(f"不支持的深度思考版本: {deep_mind}")

            sess_repo = KbotMdChatSessionRepository()
            
            # 1. 根据session_id从Oracle中获取问答pair
            session_data = await sess_repo.get_session(session_id=form.session_id)

            # 构建参考文献列表
            references = []
            host = os.getenv("KBOT_IP", "localhost")
            port = os.getenv("KBOT_PORT", "8000")
            url = f"http://{host}:{port}"
            
            if kb_results and len(kb_results) > 0:
                # 获取到引用结果
                for kb_result in kb_results:
                    reference = Reference(
                        chunk_type=kb_result.chunk_type,
                        chunk_file_path=kb_result.chunk_file_path or "",  # 确保有默认值
                        page_num=kb_result.page_num,
                        content=kb_result.content,
                        download_link=f"{url}/api/kb/download?file_id={kb_result.file_id}",
                        preview_link=f"{url}/api/kb/preview?file_id={kb_result.file_id}&page_num={kb_result.page_num}",
                        similarity_score=kb_result.similarity,
                        reranker_score=kb_result.reranker_score
                    )
                    references.append(reference)
            
            # 创建一个默认的向量（例如全零向量），避免空向量错误
            # 根据您的向量维度设置合适的默认值
            default_embedding = [0.0] * 2  # Oracle 向量不允许为空，先设置一个默认值
            
            # 构建 QAData 对象
            qa_data = QAData(
                question=form.question,
                answer="",  # 初始为空答案
                qa_embedding=default_embedding,  # 使用默认向量，避免空向量错误
                references=references,
                feedback=0,
                by=form.by,
                request_time=form.request_time,
                response_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")  # 添加微秒
            )

            if session_data:
                # 非第一次提问
                # 将问答对追加存入Oracle
                r = await sess_repo.add_qa_data(session_id=form.session_id, qa_data=qa_data)
            else:
                # 第一次提问
                # 构建 KbotBizChatSession 对象（注意：不是 KbotMdChatSession）
                session_data = KbotBizChatSession(
                    session_id=form.session_id, 
                    agent_id=form.agent_id, 
                    qa_data=[qa_data]  # 传入 QAData 对象列表
                )
                
                # 将完整session数据写入Oracle
                r = await sess_repo.create_session(session_data)

            if r:
                logger.debug(f"成功写入会话数据，session id: {form.session_id}")
            else:
                logger.warning(f"写入会话数据失败，session id: {form.session_id}")
            
            # 返回字典格式的数据给前端
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
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "*",
                }
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
        
        # 2. 根据agent_id获取agent配置
        agent_repo = KbotMdAgentRepository()
        agent = await agent_repo.get_by_id(agent_id)
        
        if agent is None:
            logger.warning("智能体不存在")
            return None
        
        model_id = agent.llm_id
        if model_id is None:
            logger.warning("智能体配置的 LLM 模型不存在")
            return None
            
        prompt_id = agent.prompt_id
        model_params = agent.llm_params if agent.llm_params else {}
        model_params["stream"] = True
        
        # 3. 根据 prompt_id 获取提示词
        prompt_template = ""
        prompt_content = await KbotMdPromptRepository().get_prompt_by_id(prompt_id) # type: ignore
        if prompt_content is None:
            prompt_template = "根据参考内容回答问题。\n\n参考内容:{context}\n\n回答的问题:{question}"
        else:
            prompt_template = prompt_content
            
        # 4. 构建上下文和问题
        context = ""
        for ref in refs:
            context += f"{ref['content']}\n"

        question = last_qa_data["question"]
        prompt = prompt_template.format(context=context.strip(), question=question) # type: ignore
        
        return last_qa_data, agent, prompt, model_id, model_params, agent_repo
    
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
                self._write_Oracle(session_id, full_response),
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
    
    async def _write_Oracle(self, session_id: str, answer: str) -> None:
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
                            #   request: Request,
                            #   background_tasks: BackgroundTasks,
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
            
            # # 1. 获取智能体信息
            # agent_repo = KbotMdAgentRepository()
            # agent_info = await agent_repo.get_by_id(agent_id)
            
            # if agent_info is None:
            #     logger.warning("智能体不存在")
            #     return None
            
            # model_id = agent_info.llm_id
            # if model_id is None:
            #     logger.warning("智能体配置的 LLM 模型不存在")
            #     return None
            
            # # 2. 获取 LLM 配置参数
            # prompt_id = agent_info.prompt_id
            # model_params = agent_info.llm_params if agent_info.llm_params else {}
            # model_params["stream"] = True
            
            # # 3. 根据 prompt_id 获取提示词
            # prompt_template = ""
            # prompt_content = await KbotMdPromptRepository().get_prompt_by_id(prompt_id)
            # if prompt_content is None:
            #     prompt_template = "根据参考内容回答问题。\n\n参考内容:{context}\n\n回答的问题:{question}"
            # else:
            #     prompt_template = prompt_content
            
            # 4. 查询向量库获取相关文档
            agent = MCPAgent(agent_id=agent_id, security=9, tags=[]) # security_level 9 表示不校验安全等级
            results  = await agent.chat(question=question, topk=topk, score_threshold=score_threshold)
            # 目前只处理知识库结果
            kb_results = []
            for result in results:
                if result.kb_results:
                    kb_results.extend(result.kb_results)
        
            
            # 5. 构建参考文献列表和上下文
            references = []
            # context = ""
            records = []
            host = os.getenv("KBOT_IP", "localhost")
            port = os.getenv("KBOT_PORT", "8000")
            url = f"http://{host}:{port}"
            
            if kb_results and len(kb_results) > 0:
                # 获取到引用结果
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

                    # 构建 dify 返回结果
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
                    

            # 8. 构建 QAData 对象
            default_embedding = [0.0] * 2  # Oracle 向量不允许为空，先设置一个默认值
            qa_data = QAData(
                question=question,
                answer="",  # 初始为空答案
                qa_embedding=default_embedding, # 先设置一个默认值，防止 Oracle 向量为空
                references=references,
                feedback=0,
                by="dify",
                request_time=request_time,
                response_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")  # 添加微秒
            )

            # 9. 将完整session数据写入Oracle
            session_data = KbotBizChatSession(
                    session_id=session_id, 
                    agent_id=agent_id, 
                    qa_data=[qa_data]  # 传入 QAData 对象列表
                )
                
            sess_repo = KbotMdChatSessionRepository()
            try:
                r = await sess_repo.create_session(session_data)
                if r:
                    logger.info(f"Oracle 写入会话数据成功")
                else:
                    logger.error(f"Oracle 写入会话数据失败")
            except Exception as e:
                logger.error(f"Oracle 写入会话数据失败: {e}")

            # # 10. 调用 LLM 获取回答
            # async def generate_stream():
            #     chunks = []
            #     try:
            #         # 调用模型生成流
            #         async for chunk in CallModel().call_llm_model(model_id, prompt, **model_params):
            #             # 检查客户端是否断开连接
            #             if await request.is_disconnected():
            #                 logger.info("检测到客户端断开连接，结束流")
            #                 break
                            
            #             yield chunk
                        
            #             # 收集内容块
            #             await self._collect_chunks(chunk, chunks)
                            
            #     except Exception as e:
            #         logger.error(f"流生成错误: {e}")
            #     finally:
            #         # 无论流如何结束，都执行清理
            #         if chunks:
            #             logger.info(f"收集到 {len(chunks)} 个内容块，执行后台写入")
            #             background_tasks.add_task(
            #                 self._process_final_response,
            #                 chunks, session_id, qa_data, agent, agent_repo
            #             )
            
            # return StreamingResponse(
            #     generate_stream(),
            #     media_type="text/event-stream",
            #     headers={
            #         "Cache-Control": "no-cache",
            #         "Connection": "keep-alive",
            #         "Access-Control-Allow-Origin": "*",
            #         "Access-Control-Allow-Headers": "*",
            #     }
            # )

            #11 返回结果

            return {"records": records}
            


        except Exception as e:
            logger.error(f"Agent chat dify failed: {e}")
            raise e

# 创建控制器实例
agent_controller = AgentController()