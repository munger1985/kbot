from langchain_community.embeddings import CohereEmbeddings, HuggingFaceEmbeddings
from sympy import EX
import config
import os
import tempfile
import pytz
from datetime import datetime
import re
from urlextract import URLExtract
from loguru import logger
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import shutil


# 获取指定目录的所有文件
def get_all_files_in_directory(directory: str):
    all_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    return all_files


# 获取指定根目录下所有子目录的所有文件
def list_files_by_level(all_files, root_dir, level=0):
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isfile(item_path):
            # logger.info(f"{'  ' * level}{item}")  # 使用缩进来表示目录级别
            # logger.info("item_file:",item_path)
            all_files.append(item_path)
        elif os.path.isdir(item_path):
            # logger.info(f"{'  ' * level}[{item}]")  # 使用方括号表示目录
            # logger.info("item_dir:",item_path)
            list_files_by_level(all_files, item_path, level + 1)  # 递归调用，增加目录级别


###删除文件
def delete_folder(folder_path):
    if os.path.isfile(folder_path):  # 判断路径是否为文件
        os.remove(folder_path)
    elif os.path.isdir(folder_path):  # 判断路径是否为文件夹
        shutil.rmtree(folder_path)


##批次处理
def batch_process(my_docs, batch_size=16):
    """
    将列表分成指定大小的批次。
    my_docs: 输入列表
    batch_size: 批次大小
    """
    for i in range(0, len(my_docs), batch_size):
        yield my_docs[i:i + batch_size]


##判断对象是否为空
def is_empty(obj):
    if obj is None:
        return True
    if isinstance(obj, str) and not obj:
        return True
    if isinstance(obj, (list, tuple, dict, set)) and not obj:
        return True
    return False

    return dest_dir


def load_embedding_model(embedding_model_dir: str) -> HuggingFaceEmbeddings:
    embeddings: HuggingFaceEmbeddings = HuggingFaceEmbeddings(model_name=embedding_model_dir,
                                                              model_kwargs={'device': 'cuda'})
    ##embeed query testing
    # query="What's the Exadata feature?"
    # embedded_query=embeddings.embed_query(query)
    # logger.info("len(embedded_query):",len(embedded_query))
    # logger.info("embedded_query[:4]:",embedded_query[:4])
    return embeddings


def load_cohere_embedding(cohere_api_key: str = "f2tdOlbKMadK2UwfcTlAI8BjTBqQSRwvwLcoSsYG") -> HuggingFaceEmbeddings:
    # cohere_api_key="6RodbluQBIwqLTyXeJMhwKiyhEamDYmF5EUpvXhc" #zouzhongfan@gmail.com登陆
    # cohere_api_key="f2tdOlbKMadK2UwfcTlAI8BjTBqQSRwvwLcoSsYG" #zhongfan.zou@oracle.com github登陆
    ###embed-english-v2.0 (default) 4096,embed-english-light-v2.0 1024,embed-multilingual-v2.0 768
    model_name = "embed-multilingual-v2.0"
    embeddings = CohereEmbeddings(model=model_name, cohere_api_key=cohere_api_key)
    ##embeed query testing
    # query="What's the Exadata feature?"
    # embedded_query=embeddings.embed_query(query)
    # logger.info("len(embedded_query):",len(embedded_query))
    # logger.info("embedded_query[:4]:",embedded_query[:4])
    return embeddings


from typing import List, Union, Dict, Optional


class SupportedVSType:
    FAISS = 'faiss'
    MILVUS = 'milvus'
    DEFAULT = 'default'
    PG = 'pg'
    ORACLE = 'oracle'


def get_kb_path(knowledge_base_name: str):
    return os.path.join(config.KB_ROOT_PATH, knowledge_base_name)


def get_content_root(knowledge_base_name: str):
    '''
    this is root path of uploaded files
    :param knowledge_base_name:
    :return:
    '''
    return os.path.join(get_kb_path(knowledge_base_name), "content")


faissVectorStore = "faissVectorStore"


def get_url_subpath(knowledge_base_name: str):
    '''
    when uploaded from url, this is used to save those webpages
    :param knowledge_base_name:
    :return:
    '''
    content_path = os.path.join(get_kb_path(knowledge_base_name), "content")
    return os.path.join(content_path, "webpages")


def get_uploaded_file_subpath(knowledge_base_name: str):
    content_path = os.path.join(get_kb_path(knowledge_base_name), "content")
    return os.path.join(content_path, "upload")


def get_vs_path(knowledge_base_name: str):
    return os.path.join(get_kb_path(knowledge_base_name), faissVectorStore)


def get_file_path(knowledge_base_name: str, doc_name: str):
    return os.path.join(get_content_root(knowledge_base_name), doc_name)


def makeSplitter(chunck_size, chunk_overlap):
    Text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunck_size, chunk_overlap=chunk_overlap,
                                                   separators=["(?<=\. )", "(?<=\。 )", "\n", " ", ""])
    return Text_splitter


from pydantic import BaseModel
import pydantic
from typing import Callable, Generator, Any


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


class AskResponseData:
    content: str = ""
    source: str = ""
    score: float = 0.0

    def __init__(self, content, source, score):
        self.content = content
        self.source = source
        self.score = score


# 获取中国时间
def get_cur_time():
    tz_SH = pytz.timezone('Asia/Shanghai')
    datetime_SH = datetime.now(tz_SH)
    return datetime_SH.strftime("%Y-%m-%d %H:%M:%S")


from pathlib import Path


def write_object(parentDir, filename, data):
    fileDiskPath = Path(parentDir) / filename
    if not fileDiskPath.parent.exists():
        fileDiskPath.parent.mkdir(parents=True)
    try:
        with open(str(fileDiskPath), 'wb') as f:
            f.write(data.content)
    except Exception as e:
        logger.error(str(e))


##数据清洗
def doc_clean(src: str):
    pattern1 = r"Copyright © [\d{4}, ]+Oracle and/or its a[ ]+ffiliates. All rights reserved."
    src = re.sub(pattern1, r"", src)
    src = re.sub(r"\n{3,}", r"\n", src)
    src = re.sub('\s', " ", src)
    src = re.sub("\n\n", "", src)
    return src


# 对LLM返回的结果作格式化处理，以后结果的格式化处理可以统一放在这个方法里面。
def format_llm_response(llm_response: Optional[str]) -> str:
    if not llm_response:
        return llm_response
    tmp_content = llm_response
    tmp_content = re.sub("\s*http", " http", tmp_content)
    tmp_content = re.sub("(，|。|；|！|“|”|？|——|：|（|）|【|】|》|《|\{|\}|\"|\'|,|;|\|!|\(|\)|\[|\]|>|<)", " ", tmp_content)
    tmp_content = re.sub("\.com[\u4e00-\u9fa5]", ".com ", tmp_content)
    tmp_content = re.sub("\.cn[\u4e00-\u9fa5]", ".cn ", tmp_content)
    extractor = URLExtract()
    urls = extractor.find_urls(tmp_content)
    result_resp = llm_response
    for url in urls:
        result_resp = result_resp.replace(f'\\\\\\"{url}\\\\\\"', f'\\"{url}\\"')
    return result_resp


import oci

from oci.ai_speech import AIServiceSpeechClientCompositeOperations

# signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()

# ai_language_client = oci.ai_language.AIServiceLanguageClient(config={}, signer=signer)
# ai_language_client = oci.ai_language.AIServiceLanguageClient(config = ociconfig)

# ai_speech_client = oci.ai_speech.AIServiceSpeechClient(config={}, signer=signer)
ai_speech_client = None
#
# object_storage_client = oci.object_storage.ObjectStorageClient(config={}, signer=signer)
object_storage_client = None

import json


def init_oci_auth(auth_type):
    finalConfig = {}
    if auth_type == 'API_KEY':
        ociconfig = oci.config.from_file()
        finalConfig = {'config': ociconfig}

    if auth_type == 'INSTANCE_PRINCIPAL':
        signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
        finalConfig = {'config': {}, 'signer': signer}
    # if region:
    #     finalConfig.update({"region":region})
    return finalConfig


from typing import List
# def get_ocr(use_cuda: bool = True) -> "RapidOCR":
#     try:
#         from rapidocr_paddle import RapidOCR
#         ocr = RapidOCR(det_use_cuda=use_cuda, cls_use_cuda=use_cuda, rec_use_cuda=use_cuda)
#     except ImportError:
#         from rapidocr_onnxruntime import RapidOCR
#         ocr = RapidOCR()
#     return ocr
from PIL import Image
from paddleocr import PaddleOCR, draw_ocr


def ppOCR(img_path, lang="en"):
    # Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
    # 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
    # need to run only once to download and load model into memory
    ocr = PaddleOCR(use_angle_cls=True, lang=lang)
    result = ocr.ocr(img_path, cls=True)
    newarr = []
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            # print(line[1][0])
            newarr.append(line[1][0])
    res = '\n'.join(newarr)
    return res


# class RapidOCRLoader(UnstructuredFileLoader):
#     def _get_elements(self) -> List:
#         def img2text(filepath):
#             resp = ""
#             ocr = get_ocr()
#             result, _ = ocr(filepath)
#             if result:
#                 ocr_result = [line[1] for line in result]
#                 resp += "\n".join(ocr_result)
#             return resp

#         text = img2text(self.file_path)
#         from unstructured.partition.text import partition_text
#         return partition_text(text=text, **self.unstructured_kwargs)
def ociSpeechASRLoader(namespace, bucket, objectName, lang):
    '''

    Args:
        bucket:  name of bucket
        objectName:  name in bucket

    Returns:

    '''

    global ai_speech_client
    if not ai_speech_client:
        ai_speech_client = oci.ai_speech.AIServiceSpeechClient(**init_oci_auth(config.auth_type))

    aiServiceSpeechClientCompositeOperations = AIServiceSpeechClientCompositeOperations(client=ai_speech_client)
    waiter_result = aiServiceSpeechClientCompositeOperations.create_transcription_job_and_wait_for_state(
        create_transcription_job_details=oci.ai_speech.models.CreateTranscriptionJobDetails(
            compartment_id=config.compartment_id,
            input_location=oci.ai_speech.models.ObjectListInlineInputLocation(
                location_type="OBJECT_LIST_INLINE_INPUT_LOCATION",
                object_locations=[oci.ai_speech.models.ObjectLocation(
                    namespace_name=namespace,
                    bucket_name=bucket,
                    object_names=[objectName])]),
            output_location=oci.ai_speech.models.OutputLocation(
                namespace_name=namespace,
                bucket_name=bucket,
                prefix="out"),
            display_name="speechJob-displayName-Value",
            description="EXAMPLE-description-Value",
            additional_transcription_formats=[],
            model_details=oci.ai_speech.models.TranscriptionModelDetails(
                model_type="WHISPER_MEDIUM",
                domain="GENERIC",
                language_code=lang,
                # zh en
                transcription_settings=oci.ai_speech.models.TranscriptionSettings(
                    diarization=oci.ai_speech.models.Diarization(
                        is_diarization_enabled=False,
                        # number_of_speakers=2
                    ))),
            normalization=oci.ai_speech.models.TranscriptionNormalization(
                is_punctuation_enabled=True,
                # filters=[
                #     oci.ai_speech.models.ProfanityTranscriptionFilter(
                #         type="PROFANITY",
                #         mode="TAG")]
            ),
        ), wait_for_states=['SUCCEEDED', 'FAILED']
    )
    if waiter_result.data.lifecycle_state == "SUCCEEDED":
        outputPrefix = waiter_result.data.output_location.prefix
        objectName = namespace + "_" + bucket + '_' + objectName + '.json'
        objectName = outputPrefix + objectName

        # 设置桶名和对象名

        # 获取对象
        global object_storage_client
        if object_storage_client is None:
            object_storage_client = oci.object_storage.ObjectStorageClient(**init_oci_auth(config.auth_type))

        response = object_storage_client.get_object(namespace, bucket, objectName)

        # 读取对象内容
        file_content = response.data.text
        jsonData = json.loads(file_content)
        textData = jsonData['transcriptions'][0]['transcription']
        # logger.info(textData)
        return textData
    else:
        return "FAILED ASR"