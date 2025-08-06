import asyncio
import datetime
import json
from typing import AsyncGenerator
from dao.repositories.kbot_md_chat_session_repo import KbotMdChatSessionRepository
from dao.repositories.kbot_md_agent_repo import KbotMdAgentRepository
from dao.repositories.kbot_md_prompt_repo import KbotMdPromptRepository
from dao.repositories.kbot_md_models_repo import KbotMdModelsRepository
from services.chat.agent_chat import Agent
from loguru import logger
from utils.common_methods import lob_to_string
from utils.decimal_encoder import DecimalEncoder
from utils.call_models import call_llm_model
from api.schemas.agent_schema import AgentChatForm, AgentChatFeedbackForm

async def agent_chat(form: AgentChatForm) -> str | None:
    try:
        agent = Agent(agent_id=form.agent_id, security=form.security_level)
        results = await agent.chat(question=form.question)
        if results is None:
            logger.warning("No results found")
            return None
        else:
            # 根据结果构建Redis数据结构并存入Redis
            # 根据返回的KBResult构建问题参考答案
            references = []
            logger.debug(f"Got {len(results)} KB results.")

            for kb_result in results:
                metadata = json.loads(json.dumps(kb_result.chunk_metadata, cls=DecimalEncoder))
                reference = {
                    "content": kb_result.chunk_doc,
                    "doc_link": f"http://localhost:8000/download?file_id={kb_result.file_id}",
                    "similarity_score": kb_result.similarity,
                    "reranker_score": kb_result.rerank_score,
                    "page_num": metadata.get("page_num", 0),
                    "source_file_ext": metadata.get("file_ext", "")
                }
                references.append(reference)

            redis_data={"SESSION_ID": form.session_id, 
                        "AGENT_ID": form.agent_id, 
                        "QA_PAIR": [{
                            "question": form.question,
                            "answer": "",
                            "reference": references,
                            "feedback": 0,
                            "by": form.by,
                            "request_time": form.request_time,
                            "response_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }]
                        }
            sess_repo = KbotMdChatSessionRepository()
            r = await sess_repo.create_session(redis_data)
            if r:
                logger.debug(f"Successfully writed to Redis，session id: {form.session_id}")
                return form.session_id
            else:
                return None
    except Exception as e:
        raise e


async def agent_stream_chat(session_id: str) -> AsyncGenerator[str, None]:
    # 1. 根据session_id从Redis中获取问答pair
    sess_repo = KbotMdChatSessionRepository()

    logger.debug(f"正在查询Redis，session_id: {session_id}")

    last_qa_pair = await sess_repo.get_last_qa_pair(session_id)

    logger.debug(f"last_qa_pair: {last_qa_pair}")

    if last_qa_pair is None:
        logger.warning("QA_PAIR not found")
        return
    
    refs = last_qa_pair["reference"]
    agent_id = last_qa_pair["agent_id"]

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
    prompt = ""
    prompt_content = await KbotMdPromptRepository().get_prompt_by_id(prompt_id) # type: ignore
    if prompt_content is None:
        prompt = "根据参考内容回答问题。"
    else:
        prompt = await lob_to_string(prompt_content)
        
    # 4. 从返回的QA_PAIR中提取问题参考答案构建LLM提示词
    for ref in refs:
        prompt += f"\n{ref['content']} "

    # 5. 从返回的QA_PAIR中提取问题构建LLM问题
    question = last_qa_pair["question"]
    prompt += f"\n{question}"
        
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
        

async def _write_to_redis(session_id: str, chunks: list):
    try:
        # Convert all chunks to strings first
        str_chunks = []
        for chunk in chunks:
            if isinstance(chunk, bytes):
                str_chunks.append(chunk.decode("utf-8"))
            else:
                str_chunks.append(str(chunk))
        
        full_response = "".join(str_chunks) if str_chunks else ""
        
        logger.debug(f"The LLM stream answer: {full_response}")

        sess_repo = KbotMdChatSessionRepository()
        await sess_repo.update_last_qa_pair_answer(
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
    
async def agent_get_session(session_id: str) -> dict | None:
    try:
        sess_repo = KbotMdChatSessionRepository()
        # 根据session_id 和问题索引，更新redis对应的问答pair中的feedback数据
        r = await sess_repo.get_session(session_id)
        return r     
    except Exception as e:  
        raise e