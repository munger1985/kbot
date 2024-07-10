# coding = utf-8
import torch
from langchain.chains import LLMChain
import json, re, os
from typing import List, Optional, Tuple
from kb_api import get_docs_with_scores, get_vs_from_kb, get_vs_path,makeSimilarDocs
import config
from util import AskResponseData
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from prompt_api import load_prompt_from_db
from util import get_cur_time, format_llm_response
from loguru import logger
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

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
            vector_store_limit=10
            ):
    # 1.初始化配置参数
    config.prompt_name = prompt_name
    config.rerankerModel = rerankerModel
    config.reranker_topk = reranker_topk
    config.score_threshold = score_threshold
    config.vector_store_limit = vector_store_limit
    logger.info(f"##1).完成配置参数初始化:{get_cur_time()}##")

    # 2.获取向量检索结果以及rerank结果
    vector_res_arr, llm_context = makeSimilarDocs(question, kb_name)
    logger.info("##2)完成获取向量检索结果以及rerank结果。##")

    # 3.设置Prompt
    promptContent = load_prompt_from_db(prompt_name)
    if not promptContent or promptContent =="":
        raise ValueError("prompt is empty !")
    prompt = PromptTemplate(input_variables=["question", "context"], template=promptContent)
    logger.info(f"##3.完成获取prompt:{prompt}##")

    # 4.调用LLM
    logger.info("######llm invoke start time:", get_cur_time())
    llm = config.MODEL_DICT.get(model_name)
    query_llm = LLMChain(llm=llm, prompt=prompt)
    response = query_llm.invoke({"context": llm_context, "question": question})
    logger.info(f"##response:{response}##")
    logger.info(f"##4).完成LLM调用:{get_cur_time()}##")

    # 5.LLM结果出来
    vector_res_arr.insert(0, AskResponseData(response.get('text'), "llm", 1))
    # 将结果对象列表转换为JSON数组
    result_str = json.dumps(
        [{"content": p.content, "source": p.source, "score": float(p.score)} for p in vector_res_arr],
        ensure_ascii=False)
    result_list = json.loads(format_llm_response(result_str))
    logger.info(f"##5).完成LLM结果处理:{get_cur_time()}##")
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
                    history_k: int = 3
                    ):
    # 1.初始化配置参数
    config.prompt_name = prompt_name
    config.rerankerModel = rerankerModel
    config.reranker_topk = reranker_topk
    config.score_threshold = score_threshold
    config.vector_store_limit = vector_store_limit
    config.history_k = history_k
    user_model = user + "_" + kb_name + "_" + model_name
    logger.info(f"##1).完成配置参数初始化:{get_cur_time()}##")

    ##2.调用Vector database retrieval and Rerank并获取处理之后结果
    vector_res_arr, llm_context = makeSimilarDocs(question, kb_name)
    logger.info("##2).完成获取向量检索结果以及rerank结果。##")

    # 3.Initial chat history, and set user_memory
    memory = ConversationBufferWindowMemory(memory_key="chat_history", k=config.history_k, return_messages=True,
                                            output_key='output')
    user_memory.setdefault(user_model, memory)
    current_memory = user_memory.get(user_model)
    logger.info("##3).完成设置ConversationBufferWindowMemory##")

    # 4.create a New Conversation
    llm = config.MODEL_DICT.get(model_name)
    assert llm is not None, f"{model_name} not in config.py"
    # if not currentConversation:
    conversation = createConversation(llm, current_memory, prompt_name, question, llm_context)
    # current_memory.save_context(inputs={"input": f"{question}"}, outputs={"output": f'found some context: \n {llm_context} \n'})
    logger.info(f"##4).完成创建New Conversation以及获取prompt##")

    # 5.invoke LLM
    logger.info("######llm invoke start time:", get_cur_time())
    inputs = {"input": f"{question}"}
    response = conversation.invoke(inputs)
    logger.info(f"##LLM reponse:{response}")
    logger.info(f"##5).完成LLM调用:{get_cur_time()}##")

    # 6.将LLM结果以及向量数据库结果，转换为JSON数组，返回
    vector_res_arr.insert(0, AskResponseData(response.get('output'), "llm", 1))
    # reference list
    result_str = json.dumps(
        [{"content": p.content, "source": p.source, "score": float(p.score)} for p in vector_res_arr],
        ensure_ascii=False)
    result_list = json.loads(format_llm_response(result_str))
    logger.info(f"##6).完成LLM结果处理:{get_cur_time()}##")
    return result_list


##有记忆rag
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
        history_k: int = 3
):
    '''
    manually implemented conversation chain
    '''
    # 1.初始化配置参数
    config.prompt_name = prompt_name
    config.rerankerModel = rerankerModel
    config.reranker_topk = reranker_topk
    config.score_threshold = score_threshold
    config.vector_store_limit = vector_store_limit
    config.history_k = history_k
    user_model = user + "_" + kb_name + "_" + model_name
    logger.info(f"##1).完成配置参数初始化:{get_cur_time()}##\n")

    # 2.获取向量检索结果以及rerank结果
    vector_res_arr, llm_context = makeSimilarDocs(question, kb_name)
    logger.info("##2).完成获取向量检索结果以及rerank结果。##\n")

    # 3.设置ConversationBufferWindowMemory
    memory = ConversationBufferWindowMemory(memory_key="chat_history", k=config.history_k, return_messages=True,
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
    llm = config.MODEL_DICT.get(model_name)
    query_llm = LLMChain(llm=llm, prompt=prompt)
    history = current_memory.load_memory_variables({})
    # logger.info(f"##history:{history}")
    llm_context = llm_context + "\n" + str(history)
    # logger.info(f"##llm_context:{llm_context}")
    response = query_llm.invoke({"context": llm_context, "question": question})
    logger.info(f"##response:{response}##")
    # 设置LLM结果到到memory中
    current_memory.save_context(inputs={"input": question}, outputs={"output": response.get('text')})
    logger.info(f"##5).完成LLM调用:{get_cur_time()}##\n")

    # 6.LLM结果出来
    vector_res_arr.insert(0, AskResponseData(response.get('text'), "llm", 1))
    # 将结果对象列表转换为JSON数组
    result_str = json.dumps(
        [{"content": p.content, "source": p.source, "score": float(p.score)} for p in vector_res_arr],
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


def ask_llm(query, model_name: str, prompt_name: Optional[str] = None):
    if prompt_name == 'default':
        prompt = PromptTemplate(input_variables=["query"], template="{query}")
    else:
        promptContent = load_prompt_from_db(prompt_name)
        prompt = PromptTemplate(input_variables=["query"], template=promptContent)

    logger.info("#### Prompt: {}", query)

    query_llm = LLMChain(llm=config.MODEL_DICT.get(model_name), prompt=prompt)
    logger.info(f"#### chat with LLM {model_name}: ", config.MODEL_DICT.get(model_name))

    response = query_llm.invoke(query)

    response = format_result(response['text'])

    vector_res_arr: List[AskResponseData] = []
    vector_res_arr.insert(0, AskResponseData(response, "llm", 1))
    # 将结果对象列表转换为JSON数组
    result_str = json.dumps(
        [{"content": p.content, "source": p.source, "score": float(p.score)} for p in vector_res_arr],
        ensure_ascii=False)
    logger.info("###### answer: {}", result_str)
    result_list = json.loads(result_str)
    return result_list


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

    llm = config.MODEL_DICT.get(model_name)
    prompt = create_prompt_template(model_name)
    chain = (
            {"context": compression_retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    resp = chain.invoke(question)

    vector_res_arr.insert(0, AskResponseData(resp, "llm", 1))
    # 将结果对象列表转换为JSON数组
    result_str = json.dumps(
        [{"content": p.content, "score": float(p.score)} for p in vector_res_arr],
        ensure_ascii=False)
    result_list = json.loads(result_str)
    return result_list

