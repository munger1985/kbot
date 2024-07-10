from fastapi import FastAPI,Body
import  argparse,uvicorn
from kb_llm_api import   compression_rag, ask_conversational_rag, clear_disable_memory_rag,ask_rag,ask_history_rag
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
)
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from kb_api import BaseResponse, ListResponse, VectorSearchResponse, create_kb, delete_batch, delete_docs, download_doc, \
    get_kb_info, \
    get_llm_info, list_embedding_models, list_kbs, list_llms, list_vector_store_types, query_in_kb, \
    recreate_vector_store, upload_docs, upload_from_url, check_vector_store_embedding_progress, sync_kbot_records, \
    delete_kb, \
    DeleteResponse, upload_from_object_storage, upload_audio_from_object_storage, text_embedding
from typing import List
from kb_llm_api import ask_llm
from prompt_api import list_prompts, create_prompt, get_prompt, delete_prompt,update_prompt
from pydantic import BaseModel
from fastapi.responses import ORJSONResponse
#sys.path.append(os.path.dirname(os.path.dirname(__file__)))

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
logger.add("kbot.log", rotation="111 MB")



##没有记忆的RAG接口
async def with_rag(
        user: str = Body(..., description="current user", examples=['Demo']),
        ask: str = Body(..., description="query", examples=['how to manage services, add users']),
        kb_name: str = Body(..., description="knowledge base name", examples=['samples']),
        llm_model: str = Body(..., description="llm model name", examples=['ChatGLM4']),
        prompt_name: str = Body('rag_default', description="prompt name" ),
        rerankerModel :str = Body('bgeReranker' ,description='which reranker model' ),
        reranker_topk: int = Body(2 ,description='reranker_topk' ),
        score_threshold:float= Body(0.6 ,description='reranker score threshold' ),
        vector_store_limit =  Body(10 ,description='the limit of query from vector db'),
):
    status: str = "success"
    err_msg: str = ""
    data: list = []
    logger.info(f"#with_rag##user:{user},ask:{ask},kb_name:{kb_name},llm_model:{llm_model},prompt_name:{prompt_name},rerankerModel:{rerankerModel},reranker_topk:{reranker_topk},score_threshold:{score_threshold},vector_store_limit:{vector_store_limit}")
    if ask != '' and llm_model != '' and kb_name != '':
        data = ask_rag(
                        user,
                        ask,
                        kb_name,
                        llm_model,
                        prompt_name,
                        rerankerModel,
                        reranker_topk ,
                        score_threshold ,
                        vector_store_limit
                )
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
        prompt_name: str = Body('rag_default', description="prompt name" ),
        rerankerModel :str = Body('bgeReranker' ,description='which reranker model' ),
        reranker_topk: int = Body(2 ,description='reranker_topk' ),
        score_threshold: float= Body(0.6 ,description='reranker score threshold' ),
        vector_store_limit: int =  Body(10 ,description='the limit of query from vector db'),
        history_k: int = Body(3 ,description='history_k' ),
):
    status: str = "success"
    err_msg: str = ""
    data: list = []
    logger.info(f"#with_history_rag##user:{user},ask:{ask},kb_name:{kb_name},llm_model:{llm_model}, prompt_name:{prompt_name},rerankerModel:{rerankerModel},reranker_topk:{reranker_topk},score_threshold:{score_threshold},vector_store_limit:{vector_store_limit},history_k:{history_k}")
    if ask != '' and llm_model != '' and kb_name != '':
        #ask_conversational_rag,ask_history_rag
        data = ask_history_rag(
                    user,
                    ask,
                    kb_name,
                    llm_model,
                    prompt_name,
                    rerankerModel,
                    reranker_topk ,
                    score_threshold ,
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

#情况有记忆对话的历史
async def clear_disable_memory_rag_history(
        user: str = Body(..., description="current user", examples=['Simon']),
        kb_name: str = Body(..., description="kb name", examples=['Oracle DB']),
        llm_model: str = Body(..., description="llm model name", examples=['chatgpt']),
):
    logger.info(f"##user:{user} kb_name:{kb_name} llm_model:{llm_model}")
    status: str = "success"
    err_msg: str = ""
    data: list = []
    data = clear_disable_memory_rag(user,kb_name,llm_model)
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
        prompt_name: str = Body('rag_default', description="prompt name" ),
        rerankerModel :str = Body('bgeReranker' ,description='which reranker model' ),
        reranker_topk: int = Body(2 ,description='reranker_topk' ),
        score_threshold:float= Body(0.6 ,description='reranker score threshold' ),
        vector_store_limit =  Body(10 ,description='the limit of query from vector db'),
        history_k: int = Body(3 ,description='history_k' ),
):
    status: str = "success"
    err_msg: str = ""
    data: list = []
    logger.info(f"#with_conversational_RAG ask:{ask},llm_model:{llm_model},kb_name:{kb_name},user:{user},prompt_name:{prompt_name},rerankerModel:{rerankerModel},reranker_topk:{reranker_topk},score_threshold:{score_threshold},vector_store_limit:{vector_store_limit},history_k:{history_k}")
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


##这个是一次性返回LLM和向量数据库的结果。
async def with_llm(
        query: str = Body(..., description="query", examples=['how to manage services, add users']),
        llm_model: str = Body(..., description="llm model name", examples=['ChatGLM4', 'llama-2-7b-chat']),
        prompt_name: str = Body('default',   description="prompt name, will use the corresponding content of prompt",
                                              examples=['default']),
):
    logger.info("\n******** question is: {}", query)
    status: str = "success"
    err_msg: str = ""
    data: list = []
    if query != '' and llm_model != '':
        data = ask_llm(query, llm_model, prompt_name)
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
        title="Sehub LLM and Vector DB Service, 🚗KBOT",
        docs_url=None, redoc_url=None
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
             summary="delete one document"
             )(delete_docs)
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
    app.get("/knowledge_base/list_vector_store_types",
            tags=["Knowledge Base Management"],
            summary="    list_vector_store_types")(list_vector_store_types)
    app.get("/knowledge_base/list_embedding_models",
            tags=["Knowledge Base Management"],
            summary="  list_embedding_models")(list_embedding_models)
    app.get("/knowledge_base/download_doc",
            tags=["Knowledge Base Management"],
            summary="download knowledge file")(download_doc)
    app.post("/knowledge_base/sync_kbot_records",
             tags=["Knowledge Base Management"],
             # response_model=ORJSONResponse,
             summary="sync_kbot_records ")(sync_kbot_records)



    app.get("/prompt/list_prompts",
            tags=["Prompt Management"],
            summary="list all Prompts")(list_prompts)
    app.post("/prompt/add_prompt",
            tags=["Prompt Management"],
            summary="add a new Prompt")(create_prompt)
    app.post("/prompt/get_prompt",
            tags=["Prompt Management"],
            summary="get a Prompt by its name")(get_prompt)
    app.post("/prompt/delete_prompt",
            tags=["Prompt Management"],
            summary="delete a Prompt by its name")(delete_prompt)
    app.post("/prompt/update_prompt",
            tags=["Prompt Management"],
            summary="update_prompt a Prompt by its name")(update_prompt)


    app.get("/chat/list_LLMs",
            tags=["LLM"],
            summary="list all llms")(list_llms)
    app.post("/chat/text_embedding",
            tags=["LLM"],
            summary="turn text to embeddings")(text_embedding)
    app.get("/chat/get_llm_info",
            tags=["LLM"],
            summary="get_llm_info")(get_llm_info)


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

    return app


app = create_app()

class QABody(BaseModel):
    question: str
    context: str | None = ""
    answer :str = ''
from transformers import pipeline

@app.post("/chat/QAbot" , tags=["Chat"],   summary="chat with Simple QA Bot")
async def qabot(qaBody:QABody):

    # 创建一个问答pipeline，默认使用一个预训练的模型（例如distilbert-base-cased-distilled-squad）
    # 你也可以指定其他模型，如bert、albert等
    qa_pipeline = pipeline("question-answering",model='NchuNLP/Chinese-Question-Answering')

    # 准备问题和上下文文本
    context = "中国的首都是北京，它是一座拥有丰富历史和文化传统的城市。"
    question = "中国的首都是哪里？"
    # 使用pipeline进行问答
    result = qa_pipeline(question=qaBody.question, context=qaBody.context)
    qaBody.answer=result['answer']

    # 输出结果
    logger.info(f"答案: '{result['answer']}'，得分: {round(result['score'], 4)}")


    return   qaBody

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
