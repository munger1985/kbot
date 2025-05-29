import  os
os.environ['USER_AGENT'] = 'kbot'
# nltk.download('punkt_tab', quiet=True)
# nltk.download('averaged_perceptron_tagger_eng', quiet=True)
from fastapi import FastAPI, Body
import argparse, uvicorn
from kb_llm_api import compression_rag,with_llm,create_llm_stream,create_rag_stream,stream_llm,stream_rag, ask_conversational_rag, clear_disable_memory_rag, ask_rag, ask_history_rag, modify_llm_parameters
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
)
from kbot_graphrag import recommended_config, default_init, graphrag_index, graphrag_local_search, graphrag_global_search ,\
    getPromptByKB, editPromptByKB,listPrompts,graphrag_update_index, editSettingsYamlByKB, getSettingsYamlByKB, checkIndexProgress, get_latest_log
from kbot_lightrag import lightragInit, lightragConfig, lightragIndex, lightragLocalSearch, \
     lightragGlobalSearch, lightragHybridSearch, lightragCheckIndexStatus, lightragGetIndexLog, \
     lightragGetEnvByKB, lightragSetEnvByKB,lightragDeleteKB,lightragDeleteKBDoc

from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from kb_api import BaseResponse, ListResponse, VectorSearchResponse, create_kb, delete_batch, delete_docs, \
    delete_webpage, download_doc, viewer_doc, \
    get_kb_info, \
    get_llm_info, list_embedding_models, list_kbs, list_llms, list_vector_store_types, query_in_kb,dify_query_from_kb_vectordb,  \
    upload_docs, upload_from_url, sync_kbot_records, \
    delete_kb, \
    DeleteResponse, upload_from_object_storage, upload_audio_from_object_storage, text_embedding
from kb_llm_api import invoke_llm, translate,get_llm
from prompt_api import list_prompts, add_prompt, get_prompt, delete_prompt, update_prompt,init_default
init_default()
from pydantic import BaseModel


## allow cors
OPEN_CROSS_DOMAIN = True

"""
logging, usage: in your module, just use
from loguru import logger
logger.debug("That's it, beautiful and simple logging!")
logger.error("That's error, beautiful and simple logging!")

"""

### init logging
from loguru import logger

logger.add("kbot.log", rotation="111 MB",level="INFO")


##没有记忆的RAG接口
async def with_rag(
        user: str = Body(..., description="current user", examples=['Demo']),
        ask: str = Body(..., description="query", examples=['how to manage services, add users']),
        kb_name: str = Body(..., description="knowledge base name", examples=['samples']),
        llm_model: str = Body(..., description="llm model name", examples=['ChatGLM4']),
        prompt_name: str = Body('rag_default', description="prompt name"),
        rerankerModel: str = Body('bgeReranker', description='which reranker model'),
        reranker_topk: int = Body(2, description='reranker_topk'),
        score_threshold: float = Body(0.6, description='reranker score threshold'),
        vector_store_limit:int =Body(10, description='the limit of query from vector db'),
        search_type:str =Body('vector', description='the type of search. eg. vector, fulltext, hybrid'),
        summary_flag:str =Body('N', description='enable summary or not'),
):
    status: str = "success"
    err_msg: str = ""
    data: list = []
    logger.debug(
        f"#with_rag##user:{user},ask:{ask},kb_name:{kb_name},llm_model:{llm_model},prompt_name:{prompt_name},rerankerModel:{rerankerModel},reranker_topk:{reranker_topk},score_threshold:{score_threshold},vector_store_limit:{vector_store_limit},search_type:{search_type},summary_flag:{summary_flag}")
    if ask != '' and llm_model != '' and kb_name != '':
        try:
            data = ask_rag(
                user,
                ask,
                kb_name,
                llm_model,
                prompt_name,
                rerankerModel,
                reranker_topk,
                score_threshold,
                vector_store_limit,
                search_type,
                summary_flag
            )
        except Exception as e:
            status = "failed"
            err_msg = str(e)
            logger.info(f"ask_rag failed err_msg:{err_msg}")
    else:
        status = "failed"
        err_msg = "Not selected llm model and knowledge base or no questions input"
    return VectorSearchResponse(
        data=data,
        status=status,
        err_msg=err_msg
    )


##有记忆RAG
async def with_history_rag(
        user: str = Body(..., description="current user", examples=['Demo']),
        ask: str = Body(..., description="query", examples=['how to create certificate in oci']),
        kb_name: str = Body(..., description="knowledge base name", examples=['samples']),
        llm_model: str = Body(..., description="llm model name", examples=['genai']),
        prompt_name: str = Body('rag_default', description="prompt name"),
        rerankerModel: str = Body('bgeReranker', description='which reranker model'),
        reranker_topk: int = Body(2, description='reranker_topk'),
        score_threshold: float = Body(0.6, description='reranker score threshold'),
        vector_store_limit: int = Body(10, description='the limit of query from vector db'),
        history_k: int = Body(3, description='history_k'),
        search_type:str =Body('vector', description='the type of search. eg. vector, fulltext, hybrid'),
        summary_flag:str =Body('N', description='enable summary or not'),
):
    status: str = "success"
    err_msg: str = ""
    data: list = []
    logger.info(
        f"#with_history_rag##user:{user},ask:{ask},kb_name:{kb_name},llm_model:{llm_model}, prompt_name:{prompt_name},rerankerModel:{rerankerModel},reranker_topk:{reranker_topk},score_threshold:{score_threshold},vector_store_limit:{vector_store_limit},history_k:{history_k},search_type:{search_type},summary_flag:{summary_flag}")
    if ask != '' and llm_model != '' and kb_name != '':
        # ask_conversational_rag,ask_history_rag
        try:
            data = ask_history_rag(
            user,
            ask,
            kb_name,
            llm_model,
            prompt_name,
            rerankerModel,
            reranker_topk,
            score_threshold,
            vector_store_limit,
            history_k,
            search_type,
            summary_flag)
        except Exception as e:
            status = "failed"
            err_msg = str(e)
            logger.info(f"ask_history_rag failed err_msg:{err_msg}")
    else:
        status = "failed"
        err_msg = "Not selected llm model and knowledge base or no questions input"
    return VectorSearchResponse(
        data=data,
        status=status,
        err_msg=err_msg
    )


# 情况有记忆对话的历史
async def clear_disable_memory_rag_history(
        user: str = Body(..., description="current user", examples=['Simon']),
        kb_name: str = Body(..., description="kb name", examples=['Oracle DB']),
        llm_model: str = Body(..., description="llm model name", examples=['chatgpt']),
):
    logger.info(f"##user:{user} kb_name:{kb_name} llm_model:{llm_model}")
    status: str = "success"
    err_msg: str = ""
    data: list = []
    data = clear_disable_memory_rag(user, kb_name, llm_model)
    return VectorSearchResponse(
        data=data,
        status=status,
        err_msg=err_msg
    )


##这个接口用新接口（with_history_rag）替代
##此接口是支持历史对话
async def with_conversational_RAG(
        ask: str = Body(..., description="query", examples=['how to create certificate in oci']),
        llm_model: str = Body(..., description="llm model name", examples=['genai']),
        kb_name: str = Body(..., description="knowledge base name", examples=['samples']),
        user: str = Body(..., description="current user", examples=['Simon']),
        prompt_name: str = Body('rag_default', description="prompt name"),
        rerankerModel: str = Body('bgeReranker', description='which reranker model'),
        reranker_topk: int = Body(2, description='reranker_topk'),
        score_threshold: float = Body(0.6, description='reranker score threshold'),
        vector_store_limit=Body(10, description='the limit of query from vector db'),
        history_k: int = Body(3, description='history_k'),
):
    status: str = "success"
    err_msg: str = ""
    data: list = []
    logger.info(
        f"#with_conversational_RAG ask:{ask},llm_model:{llm_model},kb_name:{kb_name},user:{user},prompt_name:{prompt_name},rerankerModel:{rerankerModel},reranker_topk:{reranker_topk},score_threshold:{score_threshold},vector_store_limit:{vector_store_limit},history_k:{history_k}")
    if ask != '' and llm_model != '' and kb_name != '':
        data = ask_conversational_rag(ask,
                                      llm_model,
                                      kb_name,
                                      user,
                                      prompt_name,
                                      rerankerModel,
                                      reranker_topk,
                                      score_threshold,
                                      vector_store_limit,
                                      history_k
                                      )
    else:
        status = "failed"
        err_msg = "Not selected llm model and knowledge base or no questions input"
    return VectorSearchResponse(
        data=data,
        status=status,
        err_msg=err_msg
    )


##这个是一次性返回LLM和向量数据库的结果。
async def with_compressionDoc(
        ask: str = Body(..., description="query", examples=['how to manage services, add users']),
        llm_model: str = Body(..., description="llm model name", examples=['genai']),
        kb_name: str = Body(..., description="knowledge base name", examples=['samples']),
):
    status: str = "success"
    err_msg: str = ""
    data: list = []
    if ask != '' and llm_model != '' and kb_name != '':
        data = compression_rag(ask, llm_model, kb_name)
    else:
        status = "failed"
        err_msg = "Not selected llm model and knowledge base or no questions input"
    return VectorSearchResponse(
        data=data,
        status=status,
        err_msg=err_msg
    )


async def document():
    return RedirectResponse(url="/docs")


from fastapi.staticfiles import StaticFiles


def create_app():
    app = FastAPI(
        title="Sehub LLM and Vector DB Service, 🚗🚗KBOT",
        # docs_url=None, redoc_url=None
    )
    app.mount("/static", StaticFiles(directory="./static"), name="static")
    if OPEN_CROSS_DOMAIN:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.get("/",
            response_model=BaseResponse,
            summary="swagger 文档")(document)
    ################################  KB

    app.post("/knowledge_base/create_knowledge_base",
             tags=["Knowledge Base Management"],
             response_model=BaseResponse,
             summary="make a new kb"
             )(create_kb)
    app.post("/knowledge_base/upload_from_urls",
             tags=["Knowledge Base Management"],
             response_model=BaseResponse,
             summary="upload a file From http Url"
             )(upload_from_url)
    app.post("/knowledge_base/upload_from_object_storage",
             tags=["Knowledge Base Management"],
             response_model=BaseResponse,
             summary="upload_from_object_storage"
             )(upload_from_object_storage)
    app.post("/knowledge_base/upload_audio_from_object_storage",
             tags=["Knowledge Base Management"],
             response_model=BaseResponse,
             summary="upload_audio_from_object_storage"
             )(upload_audio_from_object_storage)
    app.post("/knowledge_base/delete_docs",
             tags=["Knowledge Base Management"],
             response_model=DeleteResponse,
             summary="delete documents from kb"
             )(delete_docs)
    app.post("/knowledge_base/delete_webpage",
             tags=["Knowledge Base Management"],
             response_model=DeleteResponse,
             summary="delete one webpage from url"
             )(delete_webpage)
    app.post("/knowledge_base/delete_batch",
             tags=["Knowledge Base Management"],
             response_model=DeleteResponse,
             summary="delete one batch"
             )(delete_batch)
    app.post("/knowledge_base/delete_kb",
             tags=["Knowledge Base Management"],
             response_model=BaseResponse,
             summary="delete one knowledge_base"
             )(delete_kb)
    app.post("/knowledge_base/upload_docs",
             tags=["Knowledge Base Management"],
             response_model=BaseResponse,
             summary="upload file to kb"
             )(upload_docs)
    app.get("/knowledge_base/list_knowledge_bases",
            tags=["Knowledge Base Management"],
            response_model=ListResponse,
            summary="list all kbs")(list_kbs)
    app.get("/knowledge_base/get_kb_info",
            tags=["Knowledge Base Management"],
            response_model=BaseResponse,
            summary="get kb info    ")(get_kb_info)
    app.post("/knowledge_base/query",
             tags=["Knowledge Base Management"],
             summary="query in VectorDB")(query_in_kb)
    app.post("/knowledge_base/retrieval",
             tags=["Knowledge Base Management"],
             summary="Integrate kbot query for Dify")(dify_query_from_kb_vectordb)
    app.get("/knowledge_base/list_vector_store_types",
            tags=["Knowledge Base Management"],
            summary="    list_vector_store_types")(list_vector_store_types)
    app.get("/knowledge_base/list_embedding_models",
            tags=["Knowledge Base Management"],
            summary="  list_embedding_models")(list_embedding_models)
    app.get("/knowledge_base/download_doc",
            tags=["Knowledge Base Management"],
            summary="download knowledge file")(download_doc)
    app.get("/knowledge_base/viewer_doc",
            tags=["Knowledge Base Management"],
            summary="view knowledge file")(viewer_doc)
    app.post("/knowledge_base/sync_kbot_records",
             tags=["Knowledge Base Management"],
             # response_model=ORJSONResponse,
             summary="sync_kbot_records ")(sync_kbot_records)

    ################################  prompt

    app.get("/prompt/list_prompts",
            tags=["Prompt Management"],
            summary="list all Prompts")(list_prompts)
    app.post("/prompt/add_prompt",
             tags=["Prompt Management"],
             summary="add a new Prompt")(add_prompt)
    app.post("/prompt/get_prompt",
             tags=["Prompt Management"],
             summary="get a Prompt by its name")(get_prompt)
    app.post("/prompt/delete_prompt",
             tags=["Prompt Management"],
             summary="delete a Prompt by its name")(delete_prompt)
    app.post("/prompt/update_prompt",
             tags=["Prompt Management"],
             summary="update_prompt a Prompt by its name")(update_prompt)

    ################################  llm

    app.get("/chat/list_LLMs",
            tags=["LLM"],
            summary="list all llms")(list_llms)

    app.post("/chat/text_embedding",
             tags=["LLM"],
             summary="turn text to embeddings")(text_embedding)
    app.post("/chat/modify_llm_parameters",
             tags=["LLM"],
             summary="modify_llm_parameters")(modify_llm_parameters)
    app.get("/chat/get_llm_info",
            tags=["LLM"],
            summary="get_llm_info")(get_llm_info)

    ################################  chat, QA
    app.get("/chat/stream_llm",
            tags=["Chat"],
            summary="stream_llm")(stream_llm)
    app.get("/chat/stream_rag",
            tags=["Chat"],
            summary="stream_rag")(stream_rag)
    app.post("/chat/with_rag",
             tags=["Chat"],
             summary="chat with kbot disable memory")(with_rag)
    app.post("/chat/with_history_rag",
             tags=["Chat"],
             summary="chat with kbot enable memory")(with_history_rag)
    app.post("/chat/with_compressionDoc",
             tags=["Chat"],
             summary="chat with compression_rag bot")(with_compressionDoc)
    app.post("/chat/with_conversational_RAG",
             tags=["Chat"],
             summary="chat with_conversational_RAG bot")(with_conversational_RAG)
    app.post("/chat/clear_disable_memory_rag_history",
             tags=["Chat"],
             summary="clear_disable_memory_rag_history for a user")(clear_disable_memory_rag_history)
    app.post("/chat/with_llm",
             tags=["Chat"],
             summary="chat with llm")(with_llm)
    app.post("/chat/create_llm_stream",
             tags=["Chat"],
             summary="create llm stream")(create_llm_stream)
    app.post("/chat/create_rag_stream",
             tags=["Chat"],
             summary="create rag stream")(create_rag_stream)
    app.post("/chat/translate",
             tags=["Chat"],
             summary="Translate with llm")(translate)
    ################################  graphrag

    app.post("/graphrag/recommended_config",
             tags=["graphrag"],
             summary="init a kb for graphrag via recommended_config ")(recommended_config)
    app.post("/graphrag/default_init",
             tags=["graphrag"],
             summary="init a kb for graphrag manually   ")(default_init)
    app.post("/graphrag/getSettingsYamlByKB",
             tags=["graphrag"],
             summary="getGraphRagSettingsYaml By a kb   for graphrag    ")(getSettingsYamlByKB)
    app.post("/graphrag/editSettingsYamlByKB",
             tags=["graphrag"],
             summary="edit a kb  graphrag settings    ")(editSettingsYamlByKB)
    app.post("/graphrag/getPromptByKB",
             tags=["graphrag"],
             summary="getPromptByKB    ")(getPromptByKB)
    app.post("/graphrag/listPrompts",
             tags=["graphrag"],
             summary="list all the prompts for this kb    ")(listPrompts)
    app.post("/graphrag/graphrag_update_index",
             tags=["graphrag"],
             summary="update the index    ")(graphrag_update_index)
    app.post("/graphrag/editPromptByKB",
             tags=["graphrag"],
             summary="editPromptByKB      ")(editPromptByKB)
    app.post("/graphrag/index",
             tags=["graphrag"],
             summary="index this kb for graphrag    ")(graphrag_index)
    app.post("/graphrag/local_search",
             tags=["graphrag"],
             summary="local_search")(graphrag_local_search)
    app.post("/graphrag/global_search",
             tags=["graphrag"],
             summary="global_search")(graphrag_global_search)
    app.post("/graphrag/checkIndexProgress",
             tags=["graphrag"],
             summary="checkIndexProgress")(checkIndexProgress)
    app.post("/graphrag/get_latest_log",
             tags=["graphrag"],
             summary="check if indexing is still in progress")(get_latest_log)

    ################################  lightrag
    app.post("/lightrag/lightrag_init",
             tags=["lightrag"],
             summary="init a kb by lightrag")(lightragInit)
    app.post("/lightrag/lightrag_config",
             tags=["lightrag"],
             summary="init a kb for lightrag via lightrag_config ")(lightragConfig)
    app.post("/lightrag/lightrag_index",
             tags=["lightrag"],
             summary="index this kb by lightrag")(lightragIndex)
    app.post("/lightrag/lightrag_local_search",
             tags=["lightrag"],
             summary="lightrag local search")(lightragLocalSearch)
    app.post("/lightrag/lightrag_global_search",
             tags=["lightrag"],
             summary="lightrag global search")(lightragGlobalSearch)
    app.post("/lightrag/lightrag_hybrid_search",
             tags=["lightrag"],
             summary="lightrag Hybrid search")(lightragHybridSearch)
    app.post("/lightrag/lightrag_checkIndexStatus",
             tags=["lightrag"],
             summary="check Index Status")(lightragCheckIndexStatus)
    app.post("/lightrag/lightrag_get_index_log",
             tags=["lightrag"],
             summary="check if indexing is still in progress")(lightragGetIndexLog)
    app.post("/lightrag/lightragGetEnvByKB",
             tags=["lightrag"],
             summary="get lightrag env for kb")(lightragGetEnvByKB)
    app.post("/lightrag/lightragSetEnvByKB",
             tags=["lightrag"],
             summary="set lightrag env for kb")(lightragSetEnvByKB)
    app.post("/lightrag/lightrag_delete_kb",
             tags=["lightrag"],
             summary="delete a kb from lightrag")(lightragDeleteKB)
    app.post("/lightrag/lightrag_delete_kb_doc",
             tags=["lightrag"],
             summary="delete a kb doc from lightrag")(lightragDeleteKBDoc)
    return app


app = create_app()


class QABody(BaseModel):
    question: str
    context: str | None = ""
    answer: str = ''


from transformers import pipeline


@app.post("/chat/QAbot", tags=["Chat"], summary="chat with Simple QA Bot")
async def qabot(qaBody: QABody):
    # 创建一个问答pipeline，默认使用一个预训练的模型（例如distilbert-base-cased-distilled-squad）
    # 你也可以指定其他模型，如bert、albert等
    qa_pipeline = pipeline("question-answering", model='NchuNLP/Chinese-Question-Answering')

    # 准备问题和上下文文本
    context = "中国的首都是北京，它是一座拥有丰富历史和文化传统的城市。"
    question = "中国的首都是哪里？"
    # 使用pipeline进行问答
    result = qa_pipeline(question=qaBody.question, context=qaBody.context)
    qaBody.answer = result['answer']

    # 输出结果
    logger.info(f"答案: '{result['answer']}'，得分: {round(result['score'], 4)}")

    return qaBody


from openaiCompatible import extendApp

app = extendApp(app)


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        # oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
    )


@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=app.title + " - ReDoc",
        redoc_js_url="/static/redoc.standalone.js",
    )


def run_api(host, port, **kwargs):
    if kwargs.get("ssl_keyfile") and kwargs.get("ssl_certfile"):
        uvicorn.run(app,
                    host=host,
                    port=port,
                    ssl_keyfile=kwargs.get("ssl_keyfile"),
                    ssl_certfile=kwargs.get("ssl_certfile"),
                    )
    else:
        uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='hub KB rest',
                                     description='About hub knowledge based apis exposed as  rest-svc,  ')
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8093)
    parser.add_argument("--ssl_keyfile", type=str)
    parser.add_argument("--ssl_certfile", type=str)
    parser.add_argument("--hf_token", type=str)

    # 初始化消息
    args = parser.parse_args()
    args_dict = vars(args)
    if args.hf_token is not None:
        from huggingface_hub import login

        login(token=args.hf_token)
    run_api(host=args.host,
            port=args.port,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            )
