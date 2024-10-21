import yaml
import asyncio
import sys
from pathlib import Path
from pydantic import BaseModel, FilePath
import pydantic
from typing import Callable, Generator, Any
from fastapi import Body
import os
import shutil
import util
from fastapi.responses import StreamingResponse, FileResponse, ORJSONResponse
from prompt_api import load_prompt_from_db
from langchain_core.prompts import PromptTemplate
import config
from langchain.chains import LLMChain
import os

import pandas as pd
import tiktoken

from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.input.loaders.dfs import (
    store_entity_semantic_embeddings,
)
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.question_gen.local_gen import LocalQuestionGen
from graphrag.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext,
)
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore
from graphrag.query.structured_search.global_search.community_context import (
    GlobalCommunityContext,
)
from graphrag.query.structured_search.global_search.search import GlobalSearch

from loguru import logger


def copy_to_dest_dir(src_dir, dest_dir):
    # 创建临时目录

    # 遍历源目录中的所有文件和子目录
    for root, dirs, files in os.walk(src_dir):
        # 对每个文件，构建目标路径并复制文件
        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(
                dest_dir, os.path.relpath(src_file, src_dir))
            dst_directory = os.path.dirname(dst_file)
            # 如果目标目录不存在，则创建
            os.makedirs(dst_directory, exist_ok=True)
            shutil.copy2(src_file, dst_file)  # 使用shutil.copy2保留原文件的元数据


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


# 读取 YAML 文件
def read_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data


# 将 Python 对象写回 YAML 文件
def write_yaml(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        yaml.dump(data, file, allow_unicode=True, default_flow_style=False)


import subprocess


def runCMD(cmd):
    # 创建子进程并运行命令
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,  # 直接得到字符串输出，而不是字节
        bufsize=1,  # 行缓冲
    )

    # 实时读取标准输出和标准错误输出
    while True:
        # 从 stdout 读取一行
        output = process.stdout.readline()
        if output:
            logger.info(output.strip())  # 实时打印输出
            yield output.strip() + "\n"

        # 检查进程是否结束
        return_code = process.poll()
        if return_code is not None:
            # 进程结束后处理剩余的 stdout 和 stderr
            for output in process.stdout.readlines():
                logger.info(output.strip())

            for output in process.stderr.readlines():
                logger.info(output.strip())

            break

    logger.info(f"Process exited with code: {return_code}")


async def run_cmd(cmd):
    # 使用 asyncio.create_subprocess_exec() 来运行命令和参数
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


def check_if_init(knowledge_base_name: str = Body(..., examples=["samples"])):
    kbPath = util.get_kb_path(knowledge_base_name)
    settingsYaml = Path(kbPath) / 'graphrag' / "settings.yaml"
    if os.path.exists(settingsYaml):
        return True
    else:
        return False


def oci_sample_init(knowledge_base_name: str = Body(..., examples=["samples"]),
                    llm_endpoint: str = Body("http://localhost:8093/v1"),
                    llm_model: str = Body("OCI-meta.llama-3.1-405b-instruct"),
                    embedding_endpoint: str = Body("http://localhost:8093/v1"),
                    embedding_model: str = Body(
                        'OCI-cohere.embed-multilingual-v3.0'),
                    llm_api_key: str = Body('a'),
                    embedding_api_key: str = Body('a'),
                    claim: bool = Body(False),
                    ) -> BaseResponse:
    kbPath = util.get_kb_path(knowledge_base_name)
    graphrag_root_path = Path(kbPath) / 'graphrag'
    graphrag_input_path = graphrag_root_path / "input"
    os.makedirs(str(graphrag_input_path), exist_ok=True)
    if not check_if_init(knowledge_base_name):
        cmd = [f"{sys.executable}", "-m", "graphrag.index",
               "--init", "--root", graphrag_root_path]
        asyncio.run(run_cmd(cmd))

    settingFile = graphrag_root_path / 'settings.yaml'

    # 读取 YAML 文件
    data = read_yaml(settingFile)
    logger.info("读取的 YAML 内容：")
    logger.info(data)
    llm = data['llm']

    llm.update({"api_base": llm_endpoint})
    # llm.update({"model": "Qwen/Qwen2-7B-Instruct"})
    llm.update({"model": llm_model})
    llm.update({"api_key": llm_api_key})
    llm.update({"model_supports_json": False})

    embeddings = data['embeddings']
    # embeddings.update({"model": "Qwen/Qwen2-7B-Instruct"})
    embedllm = embeddings['llm']
    # embedllm['model'] = 'bge-m3'
    embedllm['model'] = embedding_model
    embedllm['api_key'] = embedding_api_key
    # embedllm['api_base'] = 'http://localhost:9997/v1'
    embedllm['api_base'] = embedding_endpoint
    data['storage']['base_dir'] = 'output/artifacts'
    data['claim_extraction']['enabled'] = claim
    data['reporting']['base_dir'] = 'output/logs'
    # 将修改后的数据写回 YAML 文件
    write_yaml(data, str(settingFile))

    return BaseResponse(code=200, msg=f"successfully init graphrag for kb {knowledge_base_name}")


from fastapi import Form

from fastapi.responses import Response


def checkIndexProgress(knowledge_base_name: str = Body(..., examples=["samples"]),
                       stub: str = Body('stub', examples=["no need to input"])
                       ):
    kbPath = util.get_kb_path(knowledge_base_name)
    logfile = Path(kbPath) / 'graphrag/output/logs/indexing-engine.log'
    # if ''
    num_lines = 3
    tails = 1
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
    if 'completed successfully' in tails:
        return BaseResponse(code=200, msg=f"completed")

    else:
        return BaseResponse(code=200, msg=f"not completed")


def editSettingsYamlByKB(knowledge_base_name: str = Form(..., description="知识库名称", examples=[
    "samples"]),
                         yamlContent: str = Form("", description="graphrag settings.yaml content")):
    if yamlContent == "":
        return BaseResponse(code=400, msg=f"yaml content can not be empty")

    kbPath = util.get_kb_path(knowledge_base_name)

    settingFile = Path(kbPath) / 'graphrag' / 'settings.yaml'

    with open(str(settingFile), "w", encoding="utf-8") as file:
        # 将字符串写入文件
        yamlContent = file.write(yamlContent)

    return BaseResponse(code=200, msg=f"successfully edited graphrag settings for kb {knowledge_base_name}")


def default_init(knowledge_base_name: str = Body(..., examples=["samples"]),
                 stub: str = Body('stub', examples=["no need to input"])):
    if check_if_init(knowledge_base_name):
        return BaseResponse(code=200, msg=f"successfully init graphrag for kb {knowledge_base_name}")

    kbPath = util.get_kb_path(knowledge_base_name)
    graphrag_root_path = Path(kbPath) / 'graphrag'
    graphrag_input_path = graphrag_root_path / "input"
    os.makedirs(str(graphrag_input_path), exist_ok=True)

    cmd = [f"{sys.executable}", "-m", "graphrag.index",
           "--init", "--root", graphrag_root_path]
    asyncio.run(run_cmd(cmd))
    return BaseResponse(code=200, msg=f"successfully init graphrag for kb {knowledge_base_name}")


def getSettingsYamlByKB(knowledge_base_name: str = Body(..., examples=["samples"]),
                        stub: str = Body('stub', examples=["no need to input"])):
    kbPath = util.get_kb_path(knowledge_base_name)

    settingFile = Path(kbPath) / 'graphrag' / 'settings.yaml'

    with open(str(settingFile), "r", encoding="utf-8") as file:
        # 将字符串写入文件
        yamlContent = file.read()

    return Response(content=yamlContent, media_type="text/plain")


def getPromptByKB(knowledge_base_name: str = Body(..., examples=["samples"]),
                  promptName: str = Body('entity_extraction.txt', examples=["claim_extraction.txt",
                                                                            "community_report.txt",
                                                                            "entity_extraction.txt",
                                                                            "summarize_descriptions.txt"])):
    kbPath = util.get_kb_path(knowledge_base_name)

    promptFile = Path(kbPath) / 'graphrag' / 'prompts' / promptName

    with open(str(promptFile), "r", encoding="utf-8") as file:
        # 将字符串写入文件
        content = file.read()

    return Response(content=content, media_type="text/plain")


def editPromptByKB(knowledge_base_name: str = Body(..., examples=["samples"]),
                   promptName: str = Body('entity_extraction.txt', examples=["claim_extraction.txt",
                                                                             "community_report.txt",
                                                                             "entity_extraction.txt",
                                                                             "summarize_descriptions.txt"]),
                   content: str = Body('')):
    kbPath = util.get_kb_path(knowledge_base_name)

    promptFile = Path(kbPath) / 'graphrag' / 'prompts' / promptName
    if promptFile.is_file():
        with open(promptFile, 'w', encoding='utf-8') as file:
            file.write(content)

    return Response(content=content, media_type="text/plain")


def graphrag_index(knowledge_base_name: str = Body(..., examples=["samples"]),
                   stub: str = Body('stub', examples=["for json body, no need to input "])):
    kbPath = util.get_kb_path(knowledge_base_name)
    graphrag_root_path = Path(kbPath) / 'graphrag'

    graphrag_output_path = graphrag_root_path / "output"
    util.delete_folder(str(graphrag_output_path))

    cmd = [f"{sys.executable}", "-m", "graphrag.index", "--root", str(graphrag_root_path)]

    # 实时读取标准输出
    return StreamingResponse(runCMD(cmd), media_type="text/plain")


def graphrag_local_search(knowledge_base_name: str = Body(..., examples=["samples"]),
                          question: str = Body('question', examples=["ask the detailed part as a question "]),
                          model_name: str = Body('OCI-cohere.command-r-plus',
                                                 examples=["OCI-meta.llama-3.1-405b-instruct"]),
                          prompt_name: str = 'rag_default',
                          ):
    kbPath = util.get_kb_path(knowledge_base_name)
    graphrag_root_path = Path(kbPath) / 'graphrag'
    INPUT_DIR = graphrag_root_path / "output/artifacts"
    LANCEDB_URI = f"{INPUT_DIR}/lancedb"

    COMMUNITY_REPORT_TABLE = "create_final_community_reports"
    ENTITY_TABLE = "create_final_nodes"
    ENTITY_EMBEDDING_TABLE = "create_final_entities"
    RELATIONSHIP_TABLE = "create_final_relationships"
    COVARIATE_TABLE = "create_final_covariates"
    TEXT_UNIT_TABLE = "create_final_text_units"
    COMMUNITY_LEVEL = 2
    claimEnabled = False
    if os.path.isfile(f'{INPUT_DIR}/{COVARIATE_TABLE}.parquet'):
        claimEnabled = True

    settingFile = graphrag_root_path / 'settings.yaml'
    data = read_yaml(settingFile)

    # read nodes table to get community and degree data
    entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
    entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")

    entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)

    # load description embeddings to an in-memory lancedb vectorstore
    # to connect to a remote db, specify url and port values.
    description_embedding_store = LanceDBVectorStore(
        collection_name="entity_description_embeddings",
    )
    description_embedding_store.connect(db_uri=LANCEDB_URI)
    entity_description_embeddings = store_entity_semantic_embeddings(
        entities=entities, vectorstore=description_embedding_store
    )

    logger.info(f"Entity count: {len(entity_df)}")
    entity_df.head()
    relationship_df = pd.read_parquet(f"{INPUT_DIR}/{RELATIONSHIP_TABLE}.parquet")
    relationships = read_indexer_relationships(relationship_df)

    logger.info(f"Relationship count: {len(relationship_df)}")
    relationship_df.head()
    if claimEnabled:
        covariate_df = pd.read_parquet(f"{INPUT_DIR}/{COVARIATE_TABLE}.parquet")

        claims = read_indexer_covariates(covariate_df)

        logger.info(f"Claim records: {len(claims)}")
        covariates = {"claims": claims}

    report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
    reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)

    logger.info(f"Report records: {len(report_df)}")
    report_df.head()

    text_unit_df = pd.read_parquet(f"{INPUT_DIR}/{TEXT_UNIT_TABLE}.parquet")
    text_units = read_indexer_text_units(text_unit_df)

    logger.info(f"Text unit records: {len(text_unit_df)}")
    text_unit_df.head()
    llm = data['llm']
    api_key = llm['api_key']
    llm_model = llm['model']
    embeddings = data['embeddings']
    embeddingllm = embeddings['llm']
    embedding_model = embeddingllm['model']

    llm = ChatOpenAI(
        api_key=api_key,
        model=llm_model,
        api_base=llm['api_base'],
        api_type=OpenaiApiType.OpenAI,  # OpenaiApiType.OpenAI or OpenaiApiType.AzureOpenAI
        max_retries=20,
    )

    token_encoder = tiktoken.get_encoding("cl100k_base")

    text_embedder = OpenAIEmbedding(
        api_key=embeddingllm['api_key'],
        api_base=embeddingllm['api_base'],
        api_type=OpenaiApiType.OpenAI,
        model=embedding_model,
        deployment_name=embedding_model,
        max_retries=20,
    )
    if claimEnabled == True:
        context_builder = LocalSearchMixedContext(
            community_reports=reports,
            text_units=text_units,
            entities=entities,
            relationships=relationships,
            covariates=covariates,
            entity_text_embeddings=description_embedding_store,
            embedding_vectorstore_key=EntityVectorStoreKey.ID,
            # if the vectorstore uses entity title as ids, set this to EntityVectorStoreKey.TITLE
            text_embedder=text_embedder,
            token_encoder=token_encoder,
        )
    else:
        context_builder = LocalSearchMixedContext(
            community_reports=reports,
            text_units=text_units,
            entities=entities,
            relationships=relationships,
            entity_text_embeddings=description_embedding_store,
            embedding_vectorstore_key=EntityVectorStoreKey.ID,
            # if the vectorstore uses entity title as ids, set this to EntityVectorStoreKey.TITLE
            text_embedder=text_embedder,
            token_encoder=token_encoder,
        )

    local_context_params = {
        "text_unit_prop": 0.5,
        "community_prop": 0.1,
        "conversation_history_max_turns": 5,
        "conversation_history_user_turns_only": True,
        "top_k_mapped_entities": 10,
        "top_k_relationships": 10,
        "include_entity_rank": True,
        "include_relationship_weight": True,
        "include_community_rank": False,
        "return_candidate_context": False,
        "embedding_vectorstore_key": EntityVectorStoreKey.ID,
        # set this to EntityVectorStoreKey.TITLE if the vectorstore uses entity title as ids
        "max_tokens": 9999,
        # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
    }

    llm_params = {
        "max_tokens": 1000,
        # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 1000=1500)
        "temperature": 0.1,
    }

    search_engine = LocalSearch(
        llm=llm,
        context_builder=context_builder,
        token_encoder=token_encoder,
        llm_params=llm_params,
        context_builder_params=local_context_params,
        response_type="prioritized list",
        # multiple paragraphs  free form text describing the response type and format, can be anything, e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
    )

    context_text, context_records = search_engine.context_builder.build_context(
        query=question,
        conversation_history=[],

    )


    promptContent = load_prompt_from_db(prompt_name)
    if not promptContent or promptContent == "":
        raise ValueError("prompt is empty !")
    prompt = PromptTemplate(input_variables=["question", "context"], template=promptContent)
    logger.info(f"##3.完成获取prompt:{prompt}##")

    # 4.调用LLM
    llm = config.MODEL_DICT.get(model_name)
    query_llm = LLMChain(llm=llm, prompt=prompt)
    response = query_llm.invoke({"context": context_text, "question": question})
    logger.info(f"##response:{response}##")
    # result = search_engine.search(question)
    # logger.info(result.response)
    # logger.info(f"LLM calls: {result.llm_calls}. LLM tokens: {result.prompt_tokens}")
    result = response['text']
    return BaseResponse(code=200, data=f"{result}")


def graphrag_global_search(knowledge_base_name: str = Body(..., examples=["samples"]),
                           question: str = Body('question', examples=["ask the a global question "])):
    kbPath = util.get_kb_path(knowledge_base_name)
    graphrag_root_path = Path(kbPath) / 'graphrag'
    INPUT_DIR = graphrag_root_path / "output/artifacts"
    settingFile = graphrag_root_path / 'settings.yaml'
    data = read_yaml(settingFile)
    llm = data['llm']

    llm = ChatOpenAI(
        api_key=llm['api_key'],
        model=llm['model'],
        api_base=llm['api_base'],
        api_type=OpenaiApiType.OpenAI,  # OpenaiApiType.OpenAI or OpenaiApiType.AzureOpenAI
        max_retries=20,
    )

    token_encoder = tiktoken.get_encoding("cl100k_base")
    # INPUT_DIR = "/home/opc/ragtest/output/artifacts"

    COMMUNITY_REPORT_TABLE = "create_final_community_reports"
    ENTITY_TABLE = "create_final_nodes"
    ENTITY_EMBEDDING_TABLE = "create_final_entities"

    # community level in the Leiden community hierarchy from which we will load the community reports
    # higher value means we use reports from more fine-grained communities (at the cost of higher computation cost)
    COMMUNITY_LEVEL = 2

    entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
    report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
    entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")

    reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)
    entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)
    logger.info(f"Report records: {len(report_df)}")
    report_df.head()

    context_builder = GlobalCommunityContext(
        community_reports=reports,
        entities=entities,  # default to None if you don't want to use community weights for ranking
        token_encoder=token_encoder,
    )

    context_builder_params = {
        "use_community_summary": False,
        # False means using full community reports. True means using community short summaries.
        "shuffle_data": True,
        "include_community_rank": True,
        "min_community_rank": 0,
        "community_rank_name": "rank",
        "include_community_weight": True,
        "community_weight_name": "occurrence weight",
        "normalize_community_weight": True,
        "max_tokens": 12_000,
        # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
        "context_name": "Reports",
    }

    map_llm_params = {
        "max_tokens": 1000,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }

    reduce_llm_params = {
        "max_tokens": 2000,
        # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 1000-1500)
        "temperature": 0.1,
    }

    search_engine = GlobalSearch(
        llm=llm,
        context_builder=context_builder,
        token_encoder=token_encoder,
        max_data_tokens=12_000,
        # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
        map_llm_params=map_llm_params,
        reduce_llm_params=reduce_llm_params,
        allow_general_knowledge=False,
        # set this to True will add instruction to encourage the LLM to incorporate general knowledge in the response, which may increase hallucinations, but could be useful in some use cases.
        json_mode=False,  # set this to False if your LLM model does not support JSON mode.
        context_builder_params=context_builder_params,
        concurrent_coroutines=32,
        response_type="multiple paragraphs",
        # free form text describing the response type and format, can be anything, e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
    )

    result = search_engine.search(
        question
    )

    logger.info(f'response: {result.response}')
    logger.info(f"map_responses: {result.map_responses[0].response[0]['answer']}")
    finalAnswer = result.response if result.response else result.map_responses[0].response[0]['answer']
    logger.info(f'report context: {result.context_data["reports"]}')

    logger.info(f"LLM calls: {result.llm_calls}. LLM tokens: {result.prompt_tokens}")
    return BaseResponse(code=200, data=f"{finalAnswer}")

