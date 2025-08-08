import os
import asyncio
import datetime
import json
from typing import AsyncGenerator, Any
from dao.repositories.kbot_md_chat_session_repo import KbotMdChatSessionRepository
from dao.repositories.kbot_md_agent_repo import KbotMdAgentRepository
from dao.repositories.kbot_md_prompt_repo import KbotMdPromptRepository
from dao.repositories.kbot_md_models_repo import KbotMdModelsRepository
from services.chat.agent_chat import Agent
from loguru import logger
from utils.common_methods import lob_to_string
from utils.call_models import call_llm_model
from api.schemas.agent_schema import AgentChatForm, AgentChatFeedbackForm

async def agent_chat(form: AgentChatForm) -> dict[str, Any]:
    try:
        agent = Agent(agent_id=form.agent_id, security=form.security_level)
        results = await agent.chat(question=form.question)
        sess_repo = KbotMdChatSessionRepository()
        r = None
        redis_data = None
        logger.debug(f"form: {form}")
        if results is None:
            # 第一次提问
            redis_data={"session_id": form.session_id, 
                        "agent_id": form.agent_id, 
                        "qa_data": [{
                            "question": form.question,
                            "answer": "",
                            "qa_embedding": "",
                            "references": [],
                            "feedback": 0,
                            "by": form.by,
                            "request_time": form.request_time,
                            "response_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }]
                        }
            r = await sess_repo.create_session(redis_data)
        else:
            # 非第一次提问，追加问答对 qa_data
            references = []
            logger.debug(f"Got {len(results)} matched references.")
            host = os.getenv("KBOT_HOST", "localhost")
            port = os.getenv("KBOT_PORT", "8000")
            url = f"http://{host}:{port}"
            for kb_result in results:
                reference = {
                    "chunk_type": kb_result.chunk_type,
                    "page_num": kb_result.page_num,
                    "content": kb_result.content,
                    "download_link": f"{url}/file/download?file_id={kb_result.file_id}",
                    "preview_link": f"{url}/file/preview?file_id={kb_result.file_id}",
                    "similarity_score": kb_result.similarity,
                    "reranker_score": kb_result.reranker_score
                }
                references.append(reference)

            qa_data= {
                        "question": form.question,
                        "answer": "",
                        "qa_embedding": "",
                        "references": references,
                        "feedback": 0,
                        "by": form.by,
                        "request_time": form.request_time,
                        "response_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
            # 将新的问题和答案qa_data存入Redis        
            r = await sess_repo.add_qa_data(session_id=form.session_id, qa_data=qa_data)
            redis_data = await sess_repo.get_session(session_id=form.session_id)

        if r:
            logger.debug(f"Successfully writed to Redis，session id: {form.session_id}")
        else:
            logger.warning(f"Fail to write to Redis，session id: {form.session_id}")
        return redis_data # type: ignore
    
    except Exception as e:
        raise e


async def agent_stream_chat(session_id: str) -> AsyncGenerator[str, None]:
    # 1. 根据session_id从Redis中获取问答pair
    sess_repo = KbotMdChatSessionRepository()

    logger.debug(f"正在查询Redis，session_id: {session_id}")

    last_qa_data = await sess_repo.get_last_qa_data(session_id)

    logger.debug(f"last_qa_data: {last_qa_data}")

    if last_qa_data is None:
        logger.warning("qa_data not found")
        return
    
    refs = last_qa_data["references"]
    agent_id = last_qa_data["agent_id"]

    logger.debug(f"refs: {refs}")
    logger.debug(f"agent_id: {agent_id}")
    
    # 2. 根据agent_id获取agent配置的提示词和LLM模型
    agent_repo = KbotMdAgentRepository()
    agent = await agent_repo.get_by_id(agent_id)
    
    if agent is None:
        logger.warning("Agent not found")
        return

    prompt_id = agent.prompt_id
    model_id = agent.llm_id
    model_params = agent.llm_params if agent.llm_params else {}
    model_params["stream"] = True
    
    # 3. 根据 prompt_id 获取提示词
    prompt_template = ""
    prompt_content = await KbotMdPromptRepository().get_prompt_by_id(prompt_id) # type: ignore
    if prompt_content is None:
        prompt_template = "根据参考内容回答问题。\n\n参考内容:{context}\n\n回答的问题:{question}"
    else:
        prompt_template = prompt_content
        
    # 4. 从返回的qa_data中提取问题参考答案构建LLM提示词
    context = ""
    for ref in refs:
        context += f"{ref['content']}\n"  # 收集所有参考内容，每段后用换行分隔

    # 5. 从返回的qa_data中提取问题构建LLM问题
    question = last_qa_data["question"]

    prompt = prompt_template.format(context=context.strip(), question=question) # type: ignore
        
    # 6. 根据LLM模型ID获取LLM模型
    model_unique_name = await KbotMdModelsRepository().get_unique_name_by_id(model_id) # type: ignore
    if model_unique_name is None:
        logger.warning("LLM model not found.")
        return

    # 7. 调用LLM模型并处理流式响应
    chunks = []
    try:
        async for chunk in call_llm_model(model_unique_name, prompt, **model_params):
            # 直接传递SSE流
            yield chunk
            
            # 解析内容并收集
            if isinstance(chunk, str) and chunk.startswith('data: '):
                data = chunk[6:].strip()  # 去掉"data: "前缀
                if data != '[DONE]':
                    try:
                        json_data = json.loads(data)
                        if "choices" in json_data and len(json_data["choices"]) > 0:
                            if delta := json_data["choices"][0].get("delta"):
                                if content := delta.get("content"):
                                    chunks.append(content)
                    except json.JSONDecodeError:
                        continue
            
        # 8. 流结束后异步写入Redis
        if chunks:
            asyncio.create_task(_write_to_redis(session_id, chunks))

    except Exception as e:
        logger.error(f"Stream processing error: {str(e)}")
        raise
        

async def _write_to_redis(session_id: str, 
                          #question: str, 
                          answer: list):
    try:
        # Convert all chunks to strings first
        str_chunks = []
        for chunk in answer:
            if isinstance(chunk, bytes):
                str_chunks.append(chunk.decode("utf-8"))
            else:
                str_chunks.append(str(chunk))
        
        full_response = "".join(str_chunks) if str_chunks else ""

        # 将问题和答案作为历史上下文，转换为embedding后存入redis
        # TODO
        
        logger.debug(f"The LLM stream answer: {full_response}")

        sess_repo = KbotMdChatSessionRepository()
        await sess_repo.update_last_qa_data_answer(
            session_id,
            full_response
        )
    except Exception as e:
        logger.error(f"Error writing to Redis: {str(e)}")

async def agent_feedback(form: AgentChatFeedbackForm) -> bool:
    try:
        sess_repo = KbotMdChatSessionRepository()
        session_id = form.session_id
        idx= form.question_index
        feedback = form.feedback
        # 根据session_id 和问题索引，更新redis对应的问答pair中的feedback数据
        r = await sess_repo.update_qa_feedback(session_id, idx, feedback)
        if r:
            return True
        else:
            return False       
    except Exception as e:  
        raise e
    
async def agent_get_session(session_id: str) -> dict:
    try:
        sess_repo = KbotMdChatSessionRepository()
        # 根据session_id 和问题索引，更新redis对应的问答pair中的feedback数据
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
    

async def agent_del_session(session_id: str) -> bool:
    try:
        sess_repo = KbotMdChatSessionRepository()
        # 根据session_id删除聊天会话
        return await sess_repo.delete_session(session_id)
  
    except Exception as e:  
        raise e
    