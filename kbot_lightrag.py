import os
import asyncio
import util
import configparser
import pydantic
import datetime

from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from loguru import logger
from dotenv import load_dotenv
from pathlib import Path
from fastapi import Body, Depends, Form
from fastapi.responses import Response
from config import config

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.base import DocStatus
from lightrag.types import GPTKeywordExtractionFormat

from lightrag.api.config import (
    parse_args,
    get_default_host,
)

from lightrag.api.utils_api import (
    get_combined_auth_dependency,
    display_splash_screen,
    check_env_file,
)
from lightrag.kg.shared_storage import (
    get_namespace_data,
    get_pipeline_status_lock,
    initialize_pipeline_status,
)
from lightrag.api.routers.document_routes import (
    DocumentManager,
    run_scanning_process,
)

# 定义常量
LLM_BINDINGS = ["lollms", "ollama", "openai", "openai-ollama", "azure_openai"]
EMBEDDING_BINDINGS = ["lollms", "ollama", "openai", "azure_openai"]

# 定义一个存储 RAG 对象的类
class RAGStorage:
    def __init__(self):
        self.rag = None
        self.knowledge_base_name = None


# 创建 RAGStorage 实例
rag_storage = RAGStorage()

async def run_cmd(cmd):
    # 使用 asyncio.create_subprocess_exec() 来运行命令和参数
    logger.info(f"Running command: {' '.join(cmd)}")
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    # 实时读取标准输出
    async def read_stream(stream, callback):
        while True:
            line = await stream.readline()
            if line:
                callback(line.decode('utf-8').strip())
            else:
                break

    # 定义输出的处理函数
    def print_output(line):
        logger.info(line)

    # 并行读取 stdout 和 stderr
    await asyncio.gather(
        read_stream(process.stdout, print_output),
        read_stream(process.stderr, print_output)
    )

    # 等待子进程结束
    return_code = await process.wait()
    return return_code

class BaseResponse(BaseModel):
    code: int = pydantic.Field(200, description="API status code")
    msg: str = pydantic.Field("success", description="API status message")
    data: Any = pydantic.Field(None, description="API data")

    class Config:
        json_schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
            }
        }

def check_if_init(knowledge_base_name: str = Body(..., examples=["samples"])):
    kbPath = util.get_kb_path(knowledge_base_name)
    if os.path.exists(kbPath):
        return True
    else:
        return False
async def lightragInit(knowledge_base_name: str = Body(..., examples=["samples"]),
                 stub: str = Body('stub', examples=["no need to input"])):

    kbPath = util.get_kb_path(knowledge_base_name)
    lightrag_root_path = Path(kbPath) / 'lightrag'
    graphrag_input_path = lightrag_root_path / "input"
    os.makedirs(str(graphrag_input_path), exist_ok=True)

    cmd = [f"cp", "lightrag.env", str(lightrag_root_path)]
    print(f"##lightrag init cmd: {cmd}")
    task = asyncio.create_task(run_cmd(cmd))
    env_path = graphrag_input_path/'lightrag.env'
    load_dotenv(env_path)

    args = parse_args()
    logger.info("读取的 config.ini 内容：")
    logger.info(args)

    return BaseResponse(code=200, msg=f"Successfully init LightRAG for kb {knowledge_base_name}")

def check_if_init(knowledge_base_name: str = Body(..., examples=["samples"])):
    kbPath = util.get_kb_path(knowledge_base_name)
    settingsYaml = Path(kbPath) / 'lightrag' / "config.ini"
    if os.path.exists(settingsYaml):
        return True
    else:
        return False


async def initialize_rag(knowledge_base_name, args):
    os.environ["ORACLE_WORKSPACE"] = knowledge_base_name

    # Verify that bindings are correctly setup
    if args.llm_binding not in LLM_BINDINGS:
        raise Exception(f"llm binding {args.llm_binding} not supported")
    if args.embedding_binding not in EMBEDDING_BINDINGS:
        raise Exception(f"embedding binding {args.embedding_binding} not supported")

    # Set default hosts if not provided
    if args.llm_binding_host is None:
        args.llm_binding_host = get_default_host(args.llm_binding)
    if args.embedding_binding_host is None:
        args.embedding_binding_host = get_default_host(args.embedding_binding)

    # Create working directory if it doesn't exist
    Path(args.working_dir).mkdir(parents=True, exist_ok=True)
    if args.llm_binding == "lollms" or args.embedding_binding == "lollms":
        from lightrag.llm.lollms import lollms_model_complete, lollms_embed
    if args.llm_binding == "ollama" or args.embedding_binding == "ollama":
        from lightrag.llm.ollama import ollama_model_complete, ollama_embed
    if args.llm_binding == "openai" or args.embedding_binding == "openai":
        from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    if args.llm_binding == "azure_openai" or args.embedding_binding == "azure_openai":
        from lightrag.llm.azure_openai import (
            azure_openai_complete_if_cache,
            azure_openai_embed,
        )
    if args.llm_binding_host == "openai-ollama" or args.embedding_binding == "ollama":
        from lightrag.llm.openai import openai_complete_if_cache
        from lightrag.llm.ollama import ollama_embed

    async def openai_alike_model_complete(
        prompt,
        system_prompt=None,
        history_messages=None,
        keyword_extraction=False,
        **kwargs,
    ) -> str:
        keyword_extraction = kwargs.pop("keyword_extraction", None)
        if keyword_extraction:
            kwargs["response_format"] = GPTKeywordExtractionFormat
        if history_messages is None:
            history_messages = []
        kwargs["temperature"] = args.temperature
        return await openai_complete_if_cache(
            args.llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            base_url=args.llm_binding_host,
            api_key=args.llm_binding_api_key,
            **kwargs,
        )

    async def azure_openai_model_complete(
        prompt,
        system_prompt=None,
        history_messages=None,
        keyword_extraction=False,
        **kwargs,
    ) -> str:
        keyword_extraction = kwargs.pop("keyword_extraction", None)
        if keyword_extraction:
            kwargs["response_format"] = GPTKeywordExtractionFormat
        if history_messages is None:
            history_messages = []
        kwargs["temperature"] = args.temperature
        return await azure_openai_complete_if_cache(
            args.llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            base_url=args.llm_binding_host,
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
            **kwargs,
        )

    embedding_func = EmbeddingFunc(
        embedding_dim=args.embedding_dim,
        max_token_size=args.max_embed_tokens,
        func=lambda texts: lollms_embed(
            texts,
            embed_model=args.embedding_model,
            host=args.embedding_binding_host,
            api_key=args.embedding_binding_api_key,
        )
        if args.embedding_binding == "lollms"
        else ollama_embed(
            texts,
            embed_model=args.embedding_model,
            host=args.embedding_binding_host,
            api_key=args.embedding_binding_api_key,
        )
        if args.embedding_binding == "ollama"
        else azure_openai_embed(
            texts,
            model=args.embedding_model,  # no host is used for openai,
            api_key=args.embedding_binding_api_key,
        )
        if args.embedding_binding == "azure_openai"
        else openai_embed(
            texts,
            model=args.embedding_model,
            base_url=args.embedding_binding_host,
            api_key=args.embedding_binding_api_key,
        ),
    )

    # 提取公共参数
    common_params = {
        "working_dir": args.working_dir,
        "chunk_token_size": int(args.chunk_size),
        "chunk_overlap_token_size": int(args.chunk_overlap_size),
        "llm_model_max_async": args.max_async,
        "llm_model_max_token_size": args.max_tokens,
        "embedding_func": embedding_func,
        "kv_storage": args.kv_storage,
        "graph_storage": args.graph_storage,
        "vector_storage": args.vector_storage,
        "doc_status_storage": args.doc_status_storage,
        "vector_db_storage_cls_kwargs": {
            "cosine_better_than_threshold": args.cosine_threshold
        },
        "enable_llm_cache_for_entity_extract": args.enable_llm_cache_for_extract,
        "embedding_cache_config": {
            "enabled": True,
            "similarity_threshold": 0.25,
            "use_llm_check": False,
        },
        "namespace_prefix": args.namespace_prefix,
        "auto_manage_storages_states": False,
        "max_parallel_insert": args.max_parallel_insert,
        "addon_params": {
            "example_number": 1,
            "language": "Simplified Chinese",
            "entity_types": ["organization", "person", "geo", "event"],
            "insert_batch_size": 2,
            "knowledge_base_name": knowledge_base_name,
        }
    }

    # Initialize RAG
    try:
        if args.llm_binding in ["lollms", "ollama", "openai"]:
            llm_model_func = (
                lollms_model_complete if args.llm_binding == "lollms"
                else ollama_model_complete if args.llm_binding == "ollama"
                else openai_alike_model_complete
            )
            llm_model_kwargs = {
                "host": args.llm_binding_host,
                "timeout": args.timeout,
                "options": {"num_ctx": args.max_tokens},
                "api_key": args.llm_binding_api_key,
            } if args.llm_binding in ["lollms", "ollama"] else {}

            rag = LightRAG(
                llm_model_func=llm_model_func,
                llm_model_name=args.llm_model,
                llm_model_kwargs=llm_model_kwargs,
                **common_params
            )
        else:  # azure_openai
            rag = LightRAG(
                llm_model_func=azure_openai_model_complete,
                llm_model_name=args.llm_model,
                llm_model_kwargs={
                    "timeout": args.timeout,
                },
                **common_params
            )

        await rag.initialize_storages()
        await initialize_pipeline_status()

        return rag
    except Exception as e:
        # 处理 LightRAG 初始化可能出现的异常
        print(f"Error initializing LightRAG: {e}")
        return None
async def lightragConfig(knowledge_base_name: str = Body(..., examples=["samples"]),
                       llm_endpoint: str = Body("https://api.deepseek.com/v1"),
                       llm_model: str = Body(""),
                       embedding_endpoint: str = Body(""),
                       embedding_model: str = Body(''),
                       llm_api_key: str = Body(''),
                       embedding_api_key: str = Body(''),
                       claim: bool = Body(False),
                       ) -> BaseResponse:
    kbPath = util.get_kb_path(knowledge_base_name)
    graphrag_root_path = Path(kbPath) / 'lightrag'
    # 确保输入目录存在
    if not graphrag_root_path.exists():
        return BaseResponse(code=404, msg="knowledge_base_name directory not found")

    os.environ["ORACLE_WORKSPACE"] = knowledge_base_name

    env_path = graphrag_root_path/'lightrag.env'
    load_dotenv(env_path)
    args = parse_args()
    args.working_dir = graphrag_root_path

    if llm_endpoint:
        os.environ["LLM_BINDING_HOST"] = llm_endpoint
    if llm_model:
        os.environ["LLM_MODEL"] = llm_model
    if llm_api_key:
        os.environ["LLM_BINDING_API_KEY"] = llm_api_key

    if embedding_endpoint:
        os.environ["LLM_BINDING_HOST"] = embedding_endpoint
    if embedding_model:
        os.environ["EMBEDDING_MODEL"] = embedding_model
    if embedding_api_key:
        os.environ["EMBEDDING_BINDING_API_KEY"] = knowledge_base_name

    display_splash_screen(args)
    logger.info(f"LightRAG initialed......")

    # Create application instance directly instead of using factory function
    rag = await initialize_rag(knowledge_base_name, args)

    return BaseResponse(code=200, msg=f"Successfully configured LightRAG for kb {knowledge_base_name}",data=f"{args}")

async def lightragIndex(
    knowledge_base_name: str = Body(..., examples=["samples"]),
    stub: str = Body('stub'),
) -> BaseResponse:
    try:
        kbPath = util.get_kb_path(knowledge_base_name)
        graphrag_root_path = Path(kbPath) / 'lightrag'
        graphrag_input_path = graphrag_root_path / "input"
        # 确保输入目录存在
        if not graphrag_input_path.exists():
            return BaseResponse(code=404, msg="Input directory not found")

        os.environ["ORACLE_WORKSPACE"] = knowledge_base_name

        env_path = graphrag_root_path / 'lightrag.env'
        load_dotenv(env_path)
        args = parse_args()
        args.working_dir = graphrag_root_path

        rag = await initialize_rag(knowledge_base_name, args)

        # 初始化文档管理器
        doc_manager = DocumentManager(graphrag_input_path)

        # 创建异步处理任务
        task = asyncio.create_task(run_scanning_process(rag, doc_manager))
        task_id = str(id(task))  # 生成唯一任务ID
        return BaseResponse(
            code=200, 
            msg=f"Document processing task started for {knowledge_base_name}",
            data={"task_id": task_id}
        )

    except Exception as e:
        logger.error(f"Error starting processing task: {e}")
        return BaseResponse(code=500, msg=str(e))

async def lightragCheckIndexStatus(
    knowledge_base_name: str = Body(..., examples=["samples"]),
    stub: str = Body('stub'),
) -> BaseResponse:
    kbPath = util.get_kb_path(knowledge_base_name)
    graphrag_root_path = Path(kbPath) / 'lightrag'

    os.environ["ORACLE_WORKSPACE"] = knowledge_base_name

    env_path = graphrag_root_path / 'lightrag.env'
    load_dotenv(env_path)
    args = parse_args()
    args.working_dir = graphrag_root_path

    rag = await initialize_rag(knowledge_base_name, args)

    try:
        # 获取文档状态统计
        status_counts = await rag.doc_status.get_status_counts()
        print(f"status_counts: {status_counts}")

        # 获取各状态下的文档详情
        status_details = {}
        for status in DocStatus:
            docs = await rag.doc_status.get_docs_by_status(status)
            # 提取每个文档的详细信息
            doc_details = []
            for doc_id, doc_status in docs.items():
                status_value = doc_status.status.value if hasattr(doc_status.status, 'value') else str(doc_status.status)
                
                doc_details.append({
                    "doc_id": doc_id,
                    "content_length": doc_status.content_length,
                    "created_at": doc_status.created_at,
                    "updated_at": doc_status.updated_at,
                    "chunks_count": doc_status.chunks_count,
                    "status": status_value,
                    "file_path": doc_status.file_path
                })
            
            status_details[status.value] = {
                "count": len(docs),
                "sample_docs": doc_details  # 取前3个作为示例
            }
        
        response = {
            "overview": {
                "total_documents": sum(status_counts.values()),
                "status_distribution": status_counts
            },
            "details": status_details,
        }
        
        return BaseResponse(
            code=200,
            msg="Document status retrieved successfully",
            data=response
        )
        
    except Exception as e:
        logger.error(f"Error checking document status: {e}")
        return BaseResponse(
            code=500,
            msg=f"Failed to check document status: {str(e)}"
        )

def lightragGetIndexLog(knowledge_base_name: str = Body(..., examples=["samples"]),
                       stub: str = Body('stub', examples=["no need to input"])
                       ) -> BaseResponse:
    kbPath = util.get_kb_path(knowledge_base_name)
    logfile = Path(kbPath) / 'lightrag/output/logs/indexing-engine.log'
    # if ''
    num_lines = 3

    if not os.path.exists(str(logfile)):
        return BaseResponse(code=404, msg= "Indexing not started",data="Indexing not started")
    with open(str(logfile), 'rb') as file:
        file.seek(0, 2)  # 移动到文件末尾
        buffer = bytearray()
        pointer_location = file.tell()
        while pointer_location >= 0:
            file.seek(pointer_location)
            byte = file.read(1)
            if byte == b'\n':  # 检测换行符
                num_lines -= 1
                if num_lines == 0:
                    break
            buffer.extend(byte)
            pointer_location -= 1

        # 最终的内容是反向的，需要翻转并解码
        tails = buffer[::-1].decode('utf-8')
    return BaseResponse(code=200, data=f"{tails}")

async def lightragLocalSearch(knowledge_base_name: str = Body(..., examples=["samples"]),
                           question: str = Body('question'),
                           model_name: str = Body('qwen2.5'),
                           prompt_name: str = Body('rag_default'),
                           ) -> BaseResponse:
    kbPath = util.get_kb_path(knowledge_base_name)
    graphrag_root_path = Path(kbPath) / 'lightrag'

    env_path = graphrag_root_path / 'lightrag.env'
    load_dotenv(env_path)
    args = parse_args()
    args.working_dir = graphrag_root_path

    rag = await initialize_rag(knowledge_base_name, args)

    try:
        query_param = QueryParam(
            mode="local",
            top_k=10
        )

        # 执行查询
        result = await rag.aquery(question, param=query_param)
        
    except Exception as e:
        raise BaseResponse(code=500, detail=str(e))
        return BaseResponse(
            code=500,
            msg=f"Failed to lightrag Local Search status: {str(e)}"
        )

    return BaseResponse(code=200, data=f"{result}")

async def lightragGlobalSearch(knowledge_base_name: str = Body(..., examples=["samples"]),
                           question: str = Body('question'),
                           model_name: str = Body('qwen2.5'),
                           prompt_name: str = Body('rag_default'),
                           ) -> BaseResponse:
    kbPath = util.get_kb_path(knowledge_base_name)
    graphrag_root_path = Path(kbPath) / 'lightrag'

    os.environ["ORACLE_WORKSPACE"] = knowledge_base_name
    env_path = graphrag_root_path / 'lightrag.env'
    load_dotenv(env_path)
    args = parse_args()
    args.working_dir = graphrag_root_path

    rag = await initialize_rag(knowledge_base_name, args)

    try:
        # stream response
        result = await rag.aquery(
            question,
            param=QueryParam(mode="global",top_k=10),
        )
    except Exception as e:

        return BaseResponse(
            code=500,
            msg=f"Failed to lightrag Global Search status: {str(e)}"
        )

    return BaseResponse(code=200, data=f"{result}")

async def lightragHybridSearch(knowledge_base_name: str = Body(..., examples=["samples"]),
                           question: str = Body('question'),
                           model_name: str = Body('qwen2.5'),
                           prompt_name: str = Body('rag_default'),
                           ) -> BaseResponse:
    kbPath = util.get_kb_path(knowledge_base_name)
    graphrag_root_path = Path(kbPath) / 'lightrag'

    os.environ["ORACLE_WORKSPACE"] = knowledge_base_name
    env_path = graphrag_root_path / 'lightrag.env'
    load_dotenv(env_path)
    args = parse_args()
    args.working_dir = graphrag_root_path

    rag = await initialize_rag(knowledge_base_name, args)

    try:
        # stream response
        result = await rag.aquery(
            question,
            param=QueryParam(mode="hybrid"),
        )
    except Exception as e:
        return BaseResponse(
            code=500,
            msg=f"Failed to lightrag Hybrid Search status: {str(e)}"
        )

    return BaseResponse(code=200, data=f"{result}")

async def lightragQueryGraphNode(knowledge_base_name: str = Body(..., examples=["samples"]),
                           model_name: str = Body('qwen2.5'),
                           prompt_name: str = Body('rag_default'),
                           ):
    kbPath = util.get_kb_path(knowledge_base_name)
    graphrag_root_path = Path(kbPath) / 'lightrag'
    os.environ["ORACLE_WORKSPACE"] = knowledge_base_name
    env_path = graphrag_root_path / 'lightrag.env'
    load_dotenv(env_path)
    args = parse_args()
    args.working_dir = graphrag_root_path

    rag = await initialize_rag(knowledge_base_name, args)

    try:
        result = rag.chunk_entity_relation_graph.get_all_nodes(limit=100)
    except Exception as e:
        return BaseResponse(
            code=500,
            msg=f"Failed to check document status: {str(e)}"
        )

    return BaseResponse(code=200, data=f"{result}")

async def lightragQueryGraphEdge(knowledge_base_name: str = Body(..., examples=["samples"]),
                           model_name: str = Body('qwen2.5'),
                           prompt_name: str = Body('rag_default'),
                           ):
    kbPath = util.get_kb_path(knowledge_base_name)
    graphrag_root_path = Path(kbPath) / 'lightrag'
    os.environ["ORACLE_WORKSPACE"] = knowledge_base_name
    env_path = graphrag_root_path / 'lightrag.env'
    load_dotenv(env_path)
    args = parse_args()
    args.working_dir = graphrag_root_path

    rag = await initialize_rag(knowledge_base_name, args)
    try:
        result = rag.chunk_entity_relation_graph.get_all_edges(limit=100)
    except Exception as e:
        raise BaseResponse(status_code=500, detail=str(e))

    return BaseResponse(code=200, data=f"{result}")
class DocStatusResponse(BaseModel):
    @staticmethod
    def format_datetime(dt: Any) -> Optional[str]:
        if dt is None:
            return None
        if isinstance(dt, str):
            return dt
        return dt.isoformat()

    """Response model for document status

    Attributes:
        id: Document identifier
        content_summary: Summary of document content
        content_length: Length of document content
        status: Current processing status
        created_at: Creation timestamp (ISO format string)
        updated_at: Last update timestamp (ISO format string)
        chunks_count: Number of chunks (optional)
        error: Error message if any (optional)
        metadata: Additional metadata (optional)
    """

    id: str
    content_summary: str
    content_length: int
    status: DocStatus
    created_at: str
    updated_at: str
    chunks_count: Optional[int] = None
    error: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None

class DocsStatusesResponse(BaseModel):
    statuses: Dict[DocStatus, List[DocStatusResponse]] = {}

def lightragGetEnvByKB(knowledge_base_name: str = Body(..., examples=["samples"]),
                        stub: str = Body('stub', examples=["no need to input"])):
    kbPath = util.get_kb_path(knowledge_base_name)

    settingFile = Path(kbPath) / 'lightrag' / 'lightrag.env'
    with open(str(settingFile), "r", encoding="utf-8") as file:
        # 将字符串写入文件
        envContent = file.read()

    return Response(content=envContent, media_type="text/plain")

def lightragSetEnvByKB(knowledge_base_name: str = Form(..., examples=["samples"]),
                         envContent: str = Form("", description="lightrag lightrag.env content")):
    if envContent == "":
        return BaseResponse(code=400, msg=f"env content can not be empty")

    kbPath = util.get_kb_path(knowledge_base_name)
    graphrag_root_path = Path(kbPath) / 'lightrag'
    os.environ["ORACLE_WORKSPACE"] = knowledge_base_name

    settingFile = graphrag_root_path / 'lightrag.env'

    with open(str(settingFile), "w", encoding="utf-8") as file:
        # 将字符串写入文件
        yamlfile = file.write(envContent)

    return BaseResponse(code=200, msg=f"successfully edited lightrag settings for kb {knowledge_base_name}")

async def lightragDeleteKB(
        knowledge_base_name: str = Body(..., examples=["samples"]),
        stub: str = Body('stub'),
) -> BaseResponse:
    try:
        kbPath = util.get_kb_path(knowledge_base_name)
        graphrag_root_path = Path(kbPath) / 'lightrag'
        graphrag_input_path = graphrag_root_path / "input"
        # 确保输入目录存在
        if not graphrag_input_path.exists():
            return BaseResponse(code=404, msg="Input directory not found")

        os.environ["ORACLE_WORKSPACE"] = knowledge_base_name

        env_path = graphrag_root_path / 'lightrag.env'
        load_dotenv(env_path)
        args = parse_args()
        args.working_dir = graphrag_root_path

        rag = await initialize_rag(knowledge_base_name, args)
        if not rag:
            return BaseResponse(code=500, msg="Failed to initialize RAG")

        logger.info(f"Start dropped kb")

        # Use drop method to clear all data
        drop_tasks = []
        storages = [
            rag.text_chunks,
            rag.full_docs,
            rag.entities_vdb,
            rag.relationships_vdb,
            rag.chunks_vdb,
            rag.chunk_entity_relation_graph,
            rag.doc_status,
        ]

        for storage in storages:
            if storage is not None:
                drop_tasks.append(storage.drop())

        # Wait for all drop tasks to complete
        drop_results = await asyncio.gather(*drop_tasks, return_exceptions=True)

        # Check for errors and log results
        errors = []
        storage_success_count = 0
        storage_error_count = 0

        for i, result in enumerate(drop_results):
            storage_name = storages[i].__class__.__name__
            if isinstance(result, Exception):
                error_msg = f"Error dropping {storage_name}: {str(result)}"
                errors.append(error_msg)
                logger.error(error_msg)
                storage_error_count += 1
            else:
                logger.info(f"Successfully dropped {storage_name}")
                storage_success_count += 1

    except Exception as e:
        error_msg = f"Error clearing documents: {str(e)}"
        logger.error(error_msg)
        raise BaseResponse(code=500, msg=str(e))
    finally:
        return BaseResponse(code=200, msg=f"successfully delete {knowledge_base_name}")

async def lightragDeleteKBDoc(knowledge_base_name: str = Body(..., examples=["samples"]),
                        doc_id: str = Body('stub', examples=["doc-xxx"])) -> BaseResponse:
    try:
        kbPath = util.get_kb_path(knowledge_base_name)
        graphrag_root_path = Path(kbPath) / 'lightrag'
        os.environ["ORACLE_WORKSPACE"] = knowledge_base_name

        env_path = graphrag_root_path / 'lightrag.env'
        load_dotenv(env_path)
        args = parse_args()
        args.working_dir = graphrag_root_path

        rag = await initialize_rag(knowledge_base_name, args)
        if not rag:
            return BaseResponse(code=500, msg="Failed to initialize RAG")

        # 删除文档数据
        success = await rag.adelete_by_doc_id(doc_id)
        if not success:
            return BaseResponse(code=404, msg=f"Document {doc_id} not found")

        return BaseResponse(
            code=200,
            msg=f"Successfully deleted document {doc_id} from {knowledge_base_name}"
        )
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        return BaseResponse(
            code=500,
            msg=f"Failed to delete document: {str(e)}"
        )
