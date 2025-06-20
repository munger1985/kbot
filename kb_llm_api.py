# coding = utf-8
from fastapi import Body
from fastapi.responses import Response
from langchain.chains import LLMChain
import json, re, os
from fastapi.responses import StreamingResponse
from typing import List, Optional, Tuple
from kb_api import fulltext_search, get_docs_with_scores, get_vs_from_kb, get_vs_path, makeSimilarDocs, user_settings
from config import config
import llm_models
from util import AskResponseData, SSEIdResponse
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from prompt_api import load_prompt_from_db
from util import get_cur_time, format_llm_response, remove_special_chars
from loguru import logger
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from types import SimpleNamespace
from util import BaseResponse

response = None
user_memory = {}
conversationDict = {}


def create_prompt_template(llm):
    '''
    for rag only
    '''
    if llm == 'default':
        template = """
Answer the question based only on the following context:
{context}

Question: {question}
"""
    if llm == 'genai':
        template = """{context}
Given the information above, accurately answer this question: {question}, if not enough infomation found above, just say Don't know! 
"""
    elif llm == "llama2":
        template = """你是一个乐于助人、尊重他人、诚实的AI助手。
请参考下面的上下文内容，回答后面的问题。如果您不知道答案，就回答说不知道，不要试图编造答案。
下面是上下文：
{context}
下面是要回答的问题：{question} 。请使用中文回答这个问题，不要用英文回答。"""
    else:
        template = """{context}
Given the information above, accurately answer this question: {question}
"""
    prompt = PromptTemplate.from_template(template)
    return prompt


def format_result(res: str):
    res = res.lstrip()
    res = res.rstrip()
    res = re.sub(r'\n{2,}', r'', res)
    res = re.sub(r'\n', '  <br/>', res)
    res = re.sub(r'MS', '', res)
    ##html_text = f"<p>{formatted_text}</p>"
    return res


# 创建对话
def createConversation(llm, memory, prompt_name: str = 'rag_default', question: str = "", llm_context: str = ""):
    # 1.获取提示词
    promptContent = load_prompt_from_db(prompt_name)
    replace_system_prompt = promptContent.replace("{context}", llm_context)
    replace_system_prompt = replace_system_prompt.replace("{question}", question)
    replace_system_prompt = replace_system_prompt.replace("{", "")
    replace_system_prompt = replace_system_prompt.replace("}", "")
    # logger.info(f"#promptContent:{promptContent}")
    # logger.info(f"#replace_system_prompt:{replace_system_prompt}")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", replace_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True,
        input_key="input",
        output_key="output",
        prompt=prompt
    )
    return conversation


# clear conversational
def clear_disable_memory_rag(user: str, kb_name: str, llm_model: str):
    user_model = user + "_" + kb_name + "_" + llm_model
    user_memory.pop(user_model, None)
    logger.info(f"####clear after: len(user_memory):{len(user_memory)}")
    return [user_model + ' memory has been cleared']


##没有记忆rag
def ask_rag(user: str,
            question: str,
            kb_name: str,
            model_name: str,
            prompt_name: str = 'rag_default',
            rerankerModel: str = 'bgeReranker',
            reranker_topk: int = 2,
            score_threshold: float = 0.6,
            vector_store_limit=10,
            search_type: str = 'vector',
            summary_flag: Optional[str] = 'N',
            ):
    # 1.初始化配置参数
    settings = SimpleNamespace()
    settings.prompt_name = prompt_name
    settings.rerankerModel = rerankerModel
    settings.reranker_topk = reranker_topk
    settings.score_threshold = score_threshold
    settings.vector_store_limit = vector_store_limit
    settings.search_type = search_type
    settings.summary_flag = summary_flag
    user_settings[user] = settings

    logger.info(f"##1).完成配置参数初始化:{get_cur_time()}##")

    # 2.获取向量检索结果以及rerank结果
    vector_res_arr, llm_context = makeSimilarDocs(question, kb_name, user)
    logger.info("##2)完成获取向量检索结果以及rerank结果。##")

    # 3.设置Prompt
    promptContent = load_prompt_from_db(prompt_name)
    if not promptContent or promptContent == "":
        raise ValueError("prompt is empty !")
    prompt = PromptTemplate(input_variables=["question", "context"], template=promptContent)
    logger.info(f"  3.完成获取prompt:{prompt}##")

    # 4.调用LLM
    logger.debug(f" llm invoke start time:, {get_cur_time()}")
    llm = llm_models.MODEL_DICT.get(model_name)
    query_llm = LLMChain(llm=llm, prompt=prompt)
    response = query_llm.invoke({"context": llm_context, "question": question})
    # logger.info(f"##response.text:{response.get('text')}##")
    logger.info(f"##4).完成LLM调用:{get_cur_time()}##")

    # 5.LLM结果出来
    vector_res_arr.insert(0, AskResponseData(remove_special_chars(response.get('text')), "llm", 1, "", 1, ""))
    # 将结果对象列表转换为JSON数组
    result_str = json.dumps(
        [{"content": p.content, "source": p.source, "score": float(p.score), "source_file_ext": p.source_file_ext,
          "page_num": int(p.page_num), "viewer_source": p.viewer_source} for p in vector_res_arr],
        ensure_ascii=False)
    result_list = json.loads(format_llm_response(result_str))
    logger.info(f"result_list:{result_list}##")
    logger.info(f" 5).完成LLM结果处理:{get_cur_time()}##")
    return result_list


# 根据输入的问题，RAG（支持历史对话）返回结果
def ask_history_rag(user: str,
                    question: str,
                    kb_name: str,
                    model_name: str,
                    prompt_name: str = 'rag_default',
                    rerankerModel: str = 'bgeReranker',
                    reranker_topk: int = 2,
                    score_threshold: float = 0.6,
                    vector_store_limit: int = 10,
                    history_k: int = 3,
                    search_type='vector',
                    summary_flag: Optional[str] = 'N',
                    ):
    # 1.初始化配置参数
    settings = SimpleNamespace()
    settings.prompt_name = prompt_name
    settings.rerankerModel = rerankerModel
    settings.reranker_topk = reranker_topk
    settings.score_threshold = score_threshold
    settings.vector_store_limit = vector_store_limit
    settings.search_type = search_type
    settings.history_k = history_k
    settings.summary_flag = summary_flag
    user_settings[user] = settings
    user_memory_key = user + "_" + kb_name + "_" + model_name
    logger.info(f"##1).完成配置参数初始化:{get_cur_time()}##")

    ##2.调用Vector database retrieval and Rerank并获取处理之后结果
    vector_res_arr, llm_context = makeSimilarDocs(question, kb_name, user)
    logger.info("##2).完成获取向量检索结果以及rerank结果。##")

    # 3.Initial chat history, and set user_memory
    memory = ConversationBufferWindowMemory(memory_key="chat_history", k=settings.history_k, return_messages=True,
                                            output_key='output')
    user_memory.setdefault(user_memory_key, memory)
    current_memory = user_memory.get(user_memory_key)
    logger.info("##3).完成设置ConversationBufferWindowMemory##")

    # 4.create a New Conversation
    llm = llm_models.MODEL_DICT.get(model_name)
    assert llm is not None, f"{model_name} not in config.py"
    # if not currentConversation:
    conversation = createConversation(llm, current_memory, prompt_name, question, llm_context)
    # current_memory.save_context(inputs={"input": f"{question}"}, outputs={"output": f'found some context: \n {llm_context} \n'})
    logger.info(f"##4).完成创建New Conversation以及获取prompt##")

    # 5.invoke LLM
    logger.info("######llm invoke start time:", get_cur_time())
    inputs = {"input": f"{question}"}
    response = conversation.invoke(inputs)
    # logger.info(f"##LLM reponse:{response}")
    logger.info(f"##5).完成LLM调用:{get_cur_time()}##")

    # 6.将LLM结果以及向量数据库结果，转换为JSON数组，返回
    vector_res_arr.insert(0, AskResponseData(remove_special_chars(response.get('output')), "llm", 1, "", 1, ""))
    # print(f"####response:{response.get('output')}")
    # reference list
    result_str = json.dumps(
        [{"content": p.content, "source": p.source, "score": float(p.score), "source_file_ext": p.source_file_ext,
          "page_num": int(p.page_num), "viewer_source": p.viewer_source} for p in vector_res_arr],
        ensure_ascii=False)
    result_list = json.loads(format_llm_response(result_str))
    # print(f"##result_list:{result_list}")
    logger.info(f"##6).完成LLM结果处理:{get_cur_time()}##")
    return result_list


##有记忆rag 第二种写法，修改记忆，保留
def ask_conversational_rag(
        user: str,
        question: str,
        kb_name: str,
        model_name: str,
        prompt_name: str = 'rag_default',
        rerankerModel: str = 'bgeReranker',
        reranker_topk: int = 2,
        score_threshold: float = 0.6,
        vector_store_limit: int = 10,
        history_k: int = 3,
        search_type='vector',
        summary_flag: Optional[str] = 'N',
):
    '''
    manually implemented conversation chain
    '''
    # 1.初始化配置参数
    settings = SimpleNamespace()
    settings.prompt_name = prompt_name
    settings.rerankerModel = rerankerModel
    settings.reranker_topk = reranker_topk
    settings.score_threshold = score_threshold
    settings.vector_store_limit = vector_store_limit
    settings.search_type = search_type
    settings.history_k = history_k
    settings.summary_flag = summary_flag
    user_settings[user] = settings
    user_model = user + "_" + kb_name + "_" + model_name
    logger.info(f"##1).完成配置参数初始化:{get_cur_time()}##\n")

    # 2.获取向量检索结果以及rerank结果
    vector_res_arr, llm_context = makeSimilarDocs(question, kb_name, user)
    logger.info("##2).完成获取向量检索结果以及rerank结果。##\n")

    # 3.设置ConversationBufferWindowMemory
    memory = ConversationBufferWindowMemory(memory_key="chat_history", k=settings.history_k, return_messages=True,
                                            output_key='output')
    user_memory.setdefault(user_model, memory)
    current_memory = user_memory.get(user_model)
    logger.info("##3).完成设置ConversationBufferWindowMemory##\n")

    # 4.设置Prompt
    promptContent = load_prompt_from_db(prompt_name)
    prompt = PromptTemplate(input_variables=["question", "context"], template=promptContent)
    logger.info(f"##4).完成获取prompt:{prompt}##\n")

    # 5.调用LLM
    logger.info("######llm invoke start time:", get_cur_time())
    llm = llm_models.MODEL_DICT.get(model_name)
    query_llm = LLMChain(llm=llm, prompt=prompt)
    history = current_memory.load_memory_variables({})
    # logger.info(f"##history:{history}")
    llm_context = llm_context + "\n" + str(history)
    logger.info(f"##llm_context:{llm_context}")
    response = query_llm.invoke({"context": llm_context, "question": question})
    logger.info(f"##response:{response}##")
    # 设置LLM结果到到memory中
    current_memory.save_context(inputs={"input": question}, outputs={"output": response.get('text')})
    logger.info(f"##5).完成LLM调用:{get_cur_time()}##\n")

    # 6.LLM结果出来
    vector_res_arr.insert(0, SSEIdResponse(remove_special_chars(response.get('text')), "llm", 1, "", 1, ""))
    # 将结果对象列表转换为JSON数组
    result_str = json.dumps(
        [{"content": p.content, "source": p.source, "score": float(p.score), "source_file_ext": p.source_file_ext,
          "page_num": int(p.page_num), "viewer_source": p.viewer_source} for p in vector_res_arr],
        ensure_ascii=False)
    result_list = json.loads(result_str)
    logger.info(f"##6).完成LLM结果处理:{get_cur_time()}##\n")
    return result_list


def get_prompt():
    """
    This funtion creates prompt template for cohere and attaches placeholders for the values to be updated later

    Returns:
        string: Prompt template
    """

    SYSTEM_PROMPT = """You are a contact center bot of Changi Airport. Your name is Max. Your task is to help airport customers to provide them best customer service through answering he customer queries. Use the below given context to answer the customer queries. If there is anything that you cannot answer, or you think is inappropriate to answer, simply reply as, "Sorry, I cannot help you with that."""
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    SYSTEM_PROMPT_template = B_SYS + SYSTEM_PROMPT + E_SYS
    context_instruction_template = "CHAT HISTORY: {chat_history}\n----------\nCONTEXT: {context}\n----------\n\nInstructions:1. Answer only from the given context.\n             2: Do not generate any new content out of this context.\n             3: Your answer should not include any harmful, unethical, violent, racist, sexist, pornographic, toxic, discriminatory, blasphemous, dangerous, or illegal content.\n             4: Please ensure that your responses are socially unbiased and positive in nature.\n             5: Ensure length of the answer  is within 300 words.\n\nNow, Answer the following question: {question}\n"
    prompt_template = '<s>' + B_INST + SYSTEM_PROMPT_template + context_instruction_template + E_INST

    return prompt_template


import uuid

sse_store = {}

from kb_api import BaseResponse, ListResponse, VectorSearchResponse


##这个是一次性返回LLM和向量数据库的结果。
def with_llm(
        query: str = Body(..., description="query", examples=['how to manage services, add users']),
        llm_model: str = Body(..., description="llm model name", examples=['OCI-cohere.command-a-03-2025']),
        prompt_name: str = Body('default', description="prompt name, will use the corresponding content of prompt",
                                examples=['default']),
):
    logger.info("\n******** question is: {}", query)
    status: str = "success"
    err_msg: str = ""
    data: list = []
    if query != '' and llm_model != '':
        data = invoke_llm(query, llm_model, prompt_name)
    else:
        status = "failed"
        err_msg = "Not selected llm model and knowledge base or no questions input"
    return VectorSearchResponse(
        data=data,
        status=status,
        err_msg=err_msg
    )


def create_rag_stream(
        user: str = Body(..., description="current user", examples=['Demo']),
        question: str = Body(..., description="query", examples=['how to create certificate in oci']),
        kb_name: str = Body(..., description="knowledge base name", examples=['samples']),
        llm_model: str = Body(..., description="llm model name", examples=['OCI-meta.llama-4-scout-17b']),
        prompt_name: str = Body('rag_default', description="prompt name"),
        rerankerModel: str = Body('bgeReranker', description='which reranker model'),
        reranker_topk: int = Body(2, description='reranker_topk'),
        score_threshold: float = Body(0.2, description='reranker score threshold'),
        vector_store_limit: int = Body(10, description='the limit of query from vector db'),
        search_type: str = Body('vector', description='the type of search. eg. vector, fulltext, hybrid'),
        summary_flag: str = Body('N', description='enable summary or not'),
):
    # 1.初始化配置参数
    settings = SimpleNamespace()
    settings.prompt_name = prompt_name
    settings.rerankerModel = rerankerModel
    settings.reranker_topk = reranker_topk
    settings.score_threshold = score_threshold
    settings.vector_store_limit = vector_store_limit
    settings.search_type = search_type
    settings.summary_flag = summary_flag
    user_settings[user] = settings
    user_memory_key = user + "_" + kb_name + "_" + llm_model

    ##2.调用Vector database retrieval and Rerank并获取处理之后结果
    vector_res_arr, llm_context = makeSimilarDocs(question, kb_name, user)
    session = str(uuid.uuid4())
    promptContent = load_prompt_from_db(prompt_name)
    promptTemplate = PromptTemplate(input_variables=["question", "context"], template=promptContent)
    prompt = promptTemplate.format(question=question, context=llm_context)
    input_messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {"role": "user", "content": question},
    ]
    if user_memory_key not in user_memory:
        llm = llm_models.MODEL_DICT.get(llm_model)
        assert llm is not None, f"{llm_model} not found"

        workflow = StateGraph(state_schema=MessagesState)

        def call_model(state: MessagesState):
            response = llm.invoke(state["messages"])
            return {"messages": response}

        workflow.add_edge(START, "model")
        workflow.add_node("model", call_model)

        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)

        user_memory[user_memory_key] = app
    else:
        app = user_memory.get(user_memory_key)

    config = {"configurable": {"thread_id": user_memory_key}}

    sse_store[session] = [input_messages, config, user_memory_key, app]

    new_vector_res_arr = []
    new_vector_res_arr.insert(0,
                              {
                                  "sse_session_id": session,
                              }
                              )
    new_vector_res_arr.extend([
        {
            "content": p.content,
            "source": p.source,
            "score": float(p.score),
            "source_file_ext": p.source_file_ext,
            "page_num": int(p.page_num),
            "viewer_source": p.viewer_source
        }
        for p in vector_res_arr
    ])

    response= VectorSearchResponse(
        data=new_vector_res_arr,
        status="success",
        err_msg=""
    )
    return response


def create_llm_stream(
        query: str = Body(..., description="query", examples=['how to manage services, add users']),
        llm_model: str = Body(..., description="llm model name", examples=['ChatGLM4', 'llama-2-7b-chat']),
        prompt_name: str = Body('default', description="prompt name, will use the corresponding content of prompt",
                                examples=['default']),
):
    logger.info("\n******** question is: {}", query)
    # llm = llm_models.MODEL_DICT.get(llm_model)
    session = str(uuid.uuid4())
    if len(query) > 0:
        sse_store[session] = [query, llm_model, prompt_name]
    else:
        session = "llm not found"
    return {"sse_session_id": session}


from fastapi import Query
import asyncio
import uuid

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph


async def stream_rag(sse_session_id: str = Query(..., description="sse_session_id ",
                                                 examples=["43837fae-6d7c-43e9-9b5d-e12ac80537ea"])):
    # sse_store[session] = [input_messages,config,user_memory_key,app]

    params = sse_store[sse_session_id]
    input_messages = params[0]
    config = params[1]
    user_memory_key = params[2]
    app = params[3]

    app = user_memory.get(user_memory_key)

    # 4.create a New Conversation

    async def event_generator():

        for event in app.stream({"messages": input_messages}, config, stream_mode="messages"):
            content = event[0].content
            print(content, end="", flush=True)

            if content:
                await asyncio.sleep(0)
                yield f"event: message\ndata: {content}\n\n"
        yield f"event: end\ndata: end\n\n"
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # 针对 nginx，如果用到
    }
    response = StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=headers
    )
    return response
    # current_memory.save_context(inputs={"input": f"{question}"}, outputs={"output": f'found some context: \n {llm_context} \n'})


async def stream_llm(sse_session_id: str = Query(..., description="sse_session_id ",
                                                 examples=["43837fae-6d7c-43e9-9b5d-e12ac80537ea"])):
    params = sse_store[sse_session_id]
    query = params[0]
    model_name = params[1]
    prompt_name = params[2]

    if prompt_name == 'default':
        promptTemplate = PromptTemplate(input_variables=["query"], template="{query}")
    else:
        promptContent = load_prompt_from_db(prompt_name)
        promptTemplate = PromptTemplate(input_variables=["query"], template=promptContent)
    prompt = promptTemplate.format(query=query)
    logger.info("  Prompt: {}", query)
    llm = llm_models.MODEL_DICT.get(model_name)
    # llmModel = LLMChain(llm=llm_models.MODEL_DICT.get(model_name), prompt=prompt)
    logger.info(f"Chat with LLM {model_name}: ", llm_models.MODEL_DICT.get(model_name))

    async def event_generator():

        for chunk in llm.stream(prompt):
            content = chunk.content
            print(content, end="", flush=True)

            if content:
                await asyncio.sleep(0)
                yield f"event: message\ndata: {content}\n\n"
        yield f"event: end\ndata: end\n\n"
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # 针对 nginx，如果用到
    }
    response = StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=headers
    )
    return response


def get_llm(model_name: str):
    llm = llm_models.MODEL_DICT.get(model_name)
    return llm


def invoke_llm(query, model_name: str, prompt_name: Optional[str] = None):
    if prompt_name == 'default':
        prompt = PromptTemplate(input_variables=["query"], template="{query}")
    else:
        promptContent = load_prompt_from_db(prompt_name)
        prompt = PromptTemplate(input_variables=["query"], template=promptContent)

    logger.info("  Prompt: {}", query)

    query_llm = LLMChain(llm=llm_models.MODEL_DICT.get(model_name), prompt=prompt)
    logger.info(f"  chat with LLM {model_name}: ", llm_models.MODEL_DICT.get(model_name))

    response = query_llm.invoke(query)

    # response = format_result(response['text'])

    vector_res_arr: List[AskResponseData] = []
    vector_res_arr.insert(0, AskResponseData(remove_special_chars(response.get('text')), "llm", 1, "", 1, ""))
    # 将结果对象列表转换为JSON数组
    result_str = json.dumps(
        [{"content": p.content, "source": p.source, "score": float(p.score), "source_file_ext": p.source_file_ext,
          "page_num": int(p.page_num), "viewer_source": p.viewer_source} for p in vector_res_arr],
        ensure_ascii=False)
    # logger.info("###### answer: {}", result_str)
    result_list = json.loads(result_str)
    return result_list


def modify_llm_parameters(
        model_name: str = Body("星火大模型3.0", description="query", examples=['how to manage services, add users']),
        max_tokens: int = Body(1000, description="max tokens for this llm"),
        temperature: float = Body(0.1, description="temperature for llm")):
    llm = llm_models.MODEL_DICT.get(model_name)
    model_kwargs = llm.model_kwargs
    model_kwargs['max_tokens'] = max_tokens
    model_kwargs['temperature'] = temperature
    return BaseResponse(data=str(llm.model_kwargs))


def compression_rag(question, model_name: str, kb_name: str):
    vector_res_arr: List[AskResponseData] = []
    vector_store, _ = get_vs_from_kb(kb_name)
    retriever = vector_store.as_retriever()
    compressor = FlashrankRerank()

    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

    similar_docs_with_scores = compression_retriever.get_relevant_documents(
        question
    )
    for doc in similar_docs_with_scores:
        doc_content = doc.page_content
        doc_source = 'no source'
        doc_score = doc.metadata['relevance_score']
        if abs(doc_score) >= 0.6:
            askRes = AskResponseData(doc_content, doc_source, doc_score)
            vector_res_arr.append(askRes)

    llm = llm_models.MODEL_DICT.get(model_name)
    prompt = create_prompt_template(model_name)
    chain = (
            {"context": compression_retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    resp = chain.invoke(question)

    # LLM结果出来
    vector_res_arr.insert(0, AskResponseData(remove_special_chars(response.get('text')), "llm", 1, "", 1, ""))
    # 将结果对象列表转换为JSON数组
    result_str = json.dumps(
        [{"content": p.content, "source": p.source, "score": float(p.score), "source_file_ext": p.source_file_ext,
          "page_num": int(p.page_num), "viewer_source": p.viewer_source} for p in vector_res_arr],
        ensure_ascii=False)
    result_list = json.loads(result_str)
    return result_list


def translate(query: str = Body(..., description="query", examples=['how to manage services, add users']),
              llm_model: str = Body("OCI-meta.llama-3.1-405b-instruct", description="llm model name"),
              language: str = Body('Chinese', description="the target language")):
    translatorPromptTemplate = """
    You are a very smart translator, you can translate the texts below
    {query}
    to this language {language}, just output the translation result.
    """
    prompt = PromptTemplate(input_variables=["query", 'language'], template=translatorPromptTemplate)

    logger.info("#### Prompt: {}", query)

    query_llm = LLMChain(llm=llm_models.MODEL_DICT.get(llm_model), prompt=prompt)
    logger.info(f"#### translate with LLM {llm_model}: ", llm_models.MODEL_DICT.get(llm_model))

    response = query_llm.invoke({"language": language, "query": query})

    return Response(content=response['text'], media_type="text/plain")