import concurrent.futures
import oci
from langchain_community.vectorstores import OpenSearchVectorSearch
from contextlib import contextmanager
from sqlalchemy.ext.declarative import declarative_base, DeclarativeMeta
import time
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from sqlalchemy import Column, Integer, String, DateTime, func, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy import Float, create_engine
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from fileinput import filename
from functools import wraps
from types import SimpleNamespace
from tqdm import tqdm
from loguru import logger
import shutil
from datetime import datetime
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.document_transformers import Html2TextTransformer
from util import init_oci_auth, merge_search_results, ppOCR, copy2Graphrag
from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader, Docx2txtLoader, \
    UnstructuredWordDocumentLoader
import json
import os
import urllib
import cohere
from pydantic import BaseModel, FilePath
import pydantic
from fastapi import File, Form, Query, UploadFile
from typing import Tuple
from fastapi import Body
from typing import Callable, Generator, Any
import operator
from fastapi.responses import StreamingResponse, FileResponse, ORJSONResponse
from typing import List, Union, Dict
from langchain_community.vectorstores import FAISS
import copy
# from sympy import content
import config
import llm_keys
from pathlib import Path

from vectorDB.hybrid import oracle_fulltext_helper
from vectorDB.oracle_ai_vector_search import OracleAIVector
from vectorDB.heatwave_vectorstore import HeatWaveVS
from vectorDB.oracle_vectorstore_adb import OracleAdbVS
from vectorDB.oracle_vectorstore_basedb import OracleBaseDbVS
from util import get_content_root, get_file_path, get_kb_path, get_vs_path, makeSplitter, AskResponseData, \
    get_url_subpath, \
    get_uploaded_file_subpath, delete_folder, write_object, doc_clean, ociSpeechASRLoader, \
    get_cur_time, format_llm_response
from FlagEmbedding import FlagReranker
from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter

EMBEDDING_MODEL = "e5_large_v2"
user_settings = {}


def score_threshold_process(score_threshold, k, docs):
    if score_threshold is not None:
        cmp = (
            operator.le
        )
        docs = [
            (doc, similarity)
            for doc, similarity in docs
            if cmp(similarity, score_threshold)
        ]
    return docs[:k]


class CheckProgressResponse(BaseModel):
    total: int = pydantic.Field(200, description="total file count")
    finished: int = pydantic.Field(200, description="finished count")
    current_document: str = pydantic.Field(
        "success", description="current file being processed")
    details: List = pydantic.Field(None, description="list of detailed info")


class DeleteResponse(BaseModel):
    code: int = pydantic.Field(200, description="API status code")
    msg: str = pydantic.Field("success", description="API status message")
    data: Dict = pydantic.Field(None, description="API Detailed info")


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


class ListResponse(BaseResponse):
    data: List[str] = pydantic.Field(...,
                                     description="List of knowledge base names")

    class Config:
        json_schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
                "data": ["bank", "medical", "OCI info"],
            }
        }


TEXT_SPLITTER_NAME = "RecursiveCharacterTextSplitter"

LOADER_DICT = {"UnstructuredHTMLLoader": ['.url'],
               "UnstructuredMarkdownLoader": ['.md'],
               "CustomJSONLoader": [".json"],
               "CSVLoader": [".csv"],
               "RapidOCRPDFLoader": [".pdf"],
               "ImageOCRLoader": ['.png', '.jpg', '.jpeg', '.bmp'],
               "UnstructuredFileLoader": ['.eml', '.msg', '.rst',
                                          '.rtf', '.txt', '.xml',
                                          '.epub', '.odt',
                                          '.ppt', '.pptx', '.tsv', ],
               "UnstructuredWordDocumentLoader": ['.docx', '.doc'],

               'OCISpeechLoader': ['.wav']
               }
SUPPORTED_EXTS = [ext for sublist in LOADER_DICT.values() for ext in sublist]


def get_LoaderClass(file_extension):
    for LoaderClass, extensions in LOADER_DICT.items():
        if file_extension in extensions:
            return LoaderClass


class KnowledgeFile:
    chunk_size: int = None
    chunk_overlap: int = None
    batch_name = './'
    status = 'success'
    msg = ''
    filename: str = None
    filepath: str = None
    knowledge_base_name: str = None
    ext: str = None
    type: str = 'file'
    full_text: str = ''

    def get_mtime(self):
        if self.ext == '.url':
            now = time.time()
            return now
        return os.path.getmtime(self.filepath)

    def get_loader(self, loader_name: str, filepath: str):
        if filepath.lower().endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif loader_name == 'UnstructuredWordDocumentLoader':
            # loader = Docx2txtLoader(filepath)
            loader = UnstructuredWordDocumentLoader(filepath)
        elif filepath.lower().endswith(".wav"):
            loader = ociSpeechASRLoader(
                self.namespace, self.bucket, self.objectName, self.lang)
        elif loader_name == 'ImageOCRLoader':
            loader = ppOCR(file_path=self.filepath)
        else:
            loader = TextLoader(filepath, autodetect_encoding=True)
        return loader

    def get_size(self):
        if self.ext == '.url':
            return 0  # url not saved to local
        return os.path.getsize(self.filepath)

    def file2docs(self):
        loader = self.get_loader(self.document_loader_name, self.filepath)
        self.docs = loader.load()
        return self.docs

    def __init__(
            self,
            # knowledge_base_name: str='knowledge_base_name',
            **kwargs,
            # filename: str='default_filename',
            # filepath:str = 'default_filepath',
            # type: str = 'file'
    ):
        '''
        对应知识库目录中的文件，必须是磁盘上存在的才能进行向量化等操作。

        '''
        # self.kb_name = kwargs.get('knowledge_base_name')
        # self.filename =kwargs.get('filename')
        for key, value in kwargs.items():
            setattr(self, key, value)
        # self.filepath =kwargs.get('filepath')
        if self.type == 'webpage':
            self.ext = '.url'
            # doc_path = get_content_root(knowledge_base_name)
            # url_subpath = os.path.join(doc_path, 'webpages')
            # self.filepath = self.filename
            # loader = WebBaseLoader(filename)
            logger.info(self.filepath)
        # elif type=='audio':
        #     filename = filename.replace('/','-')
        #     self.filepath=get_file_path(knowledge_base_name, filename)
        #     self.ext = os.path.splitext(filename)[-1].lower()
        else:
            self.ext = os.path.splitext(kwargs.get('filename'))[-1].lower() if kwargs.get('filename') else \
                os.path.splitext(kwargs.get('filepath'))[-1].lower()
            # filename = filename.replace('/', '-')
            if not kwargs.get('filepath'):
                self.kbPath = get_kb_path(kwargs.get('knowledge_base_name'))
                fileDiskPath = Path(get_content_root(
                    kwargs.get('knowledge_base_name'))) / self.filename
                if not fileDiskPath.parent.exists:
                    fileDiskPath.parent.mkdir(parents=True)
                self.filepath = str(fileDiskPath)
        if self.ext not in SUPPORTED_EXTS:
            logger.warning(f"暂未支持的文件格式 {self.ext}")
        self.docs = None
        self.texts = None
        self.document_loader_name = get_LoaderClass(self.ext)
        self.text_splitter_name = TEXT_SPLITTER_NAME

    def docs2texts(
            self,
            docs: List[Document] = None,
            text_splitter: TextSplitter = None,
    ):
        if not docs:
            return []
        new_docs: List[Document] = []
        # 对原始数据做清洗，去掉多余的空格等。
        for doc in docs:
            doc.page_content = doc_clean(doc.page_content)
            new_docs.append(doc)
        self.texts = text_splitter.split_documents(new_docs)
        return self.texts

    def url2Docs(
            self,
            text_splitter: TextSplitter = None,
    ):

        loader = WebBaseLoader(self.filepath, encoding='utf-8')
        docs = loader.load()
        docs_transformed = Html2TextTransformer().transform_documents(docs)
        texts = text_splitter.split_documents(docs_transformed)
        return texts

    def file2text(
            self,
            text_splitter: TextSplitter = None,
            lang='en'
    ):
        '''
        for uploading files from web
        :param text_splitter:
        :param lang: when there are some images or audios, specify the language
        :return: chunked langchain documents
        '''
        if self.ext == '.wav':

            # asr to text .
            #  for asr , need to upload  to oci buckets first。
            asrFilePosixPath = Path(self.kbPath) / \
                               Path('asr') / Path(self.filename + '.txt')
            if not asrFilePosixPath.parent.exists():
                asrFilePosixPath.parent.mkdir(parents=True)
            with open(str(asrFilePosixPath), "w") as f:
                f.write(self.full_text)
            document = Document(page_content=self.full_text)
            document.metadata['source'] = self.filepath
            self.docs = [document]
        elif self.document_loader_name == 'ImageOCRLoader':
            fileTextOneString = ppOCR(self.filepath, lang)
            ocrFilePosixPath = Path(self.kbPath) / \
                               Path('ocr') / Path(self.filename + '.txt')
            if not ocrFilePosixPath.parent.exists():
                ocrFilePosixPath.parent.mkdir(parents=True)
            with open(str(ocrFilePosixPath), "w") as f:
                f.write(fileTextOneString)
            document = Document(page_content=fileTextOneString)
            document.metadata['source'] = self.filepath
            self.docs = [document]
        else:
            self.docs = self.file2docs()
        self.full_text = "\n".join(doc.page_content for doc in self.docs)

        copy2Graphrag(self)

        self.texts = self.docs2texts(docs=self.docs,
                                     text_splitter=text_splitter)
        return self.texts

    def rebuild_file2text(
            self,
            text_splitter: TextSplitter = None,
    ):
        '''
        when recreating the whole vectorStore, scanning all the text files beneath the content Root dir of a certain knowledge base
        :param text_splitter:
        :return: chunked langchain documents
        '''
        if self.ext == '.wav':
            # asr to text .
            asrFilePosixPath = Path(self.kbPath) / \
                               Path('asr') / Path(self.filename + '.txt')
            with open(str(asrFilePosixPath), "r") as f:
                fileTextOneString = f.read()
            document = Document(page_content=fileTextOneString)
            document.metadata['source'] = self.filepath
            docs = [document]
        elif self.document_loader_name == 'ImageOCRLoader':
            ocrFilePosixPath = Path(self.kbPath) / \
                               Path('ocr') / Path(self.filename + '.txt')
            with open(str(ocrFilePosixPath), "r") as f:
                fileTextOneString = f.read()
            document = Document(page_content=fileTextOneString)
            document.metadata['source'] = self.filepath
            docs = [document]
        else:
            docs = self.file2docs()

        self.texts = self.docs2texts(docs=docs,
                                     text_splitter=text_splitter)
        return self.texts


if config.KB_ROOT_PATH == 'auto':
    # 获取当前脚本所在的路径
    current_file_path = os.path.abspath(__file__)
    # 获取父目录
    parent_directory = os.path.dirname(current_file_path)
    # 获取父目录的父目录
    grandparent_directory = os.path.dirname(parent_directory)
    config.KB_ROOT_PATH = Path(grandparent_directory) / 'kbroot'
    logger.info(f"KBroot Path is set in {config.KB_ROOT_PATH}")
    config.sqlite_path = config.KB_ROOT_PATH

# 创建一个 SQLite 数据库引擎
sqlitePath = os.path.expanduser(f'{config.sqlite_path}/kbdatabase.db')
parent_path = os.path.dirname(sqlitePath)
if not os.path.exists(parent_path):
    os.makedirs(parent_path)
    logger.info(
        f"------- created sqlite db file directory {parent_path} ------- ")
engine = create_engine(
    'sqlite:///' + sqlitePath,
    json_serializer=lambda obj: json.dumps(obj, ensure_ascii=False),
)

Base: DeclarativeMeta = declarative_base()


class KnowledgeBatchInfo(Base):
    __tablename__ = 'knowledge_batch_info'
    id = Column(Integer, primary_key=True,
                autoincrement=True, comment='kb batch ID')
    batch_name = Column(String, comment='batch name    ')
    kb_name = Column(String, comment='kb name')
    chunk_size = Column(Integer, default=250, comment="the size of a chunk")
    chunk_overlap = Column(Integer, default=25,
                           comment="the chunks overlapping size  ")
    update_time = Column(DateTime, default=func.now(), comment='update_time ')
    file_count = Column(Integer, default=0, comment='total file count')
    __table_args__ = (
        Index('batch_in_kb', 'batch_name', 'kb_name'),  # 复合索引
    )


class KnowledgeFileModel(Base):
    """
    知识文件模型
    """
    __tablename__ = 'knowledge_file'
    # id = Column(Integer, primary_key=True, autoincrement=True, comment='知识文件ID')
    filepath = Column(String, primary_key=True, comment='file absolute path')
    file_name = Column(String(255), comment='文件名')
    chunk_size = Column(Integer, default=250, comment="the size of a chunk")
    chunk_overlap = Column(Integer, default=25,
                           comment="the chunks overlapping size  ")
    batch_name = Column(String, comment='batch name')
    status = Column(
        String, comment='success or failure when saving it to kbot')
    msg = Column(String, comment='error message when saving it to kbot')
    file_ext = Column(String(10), comment='文件扩展名')
    kb_name = Column(String(50), comment='所属知识库名称')
    document_loader_name = Column(String(50), comment='文档加载器名称')
    text_splitter_name = Column(String(50), comment='文本分割器名称')
    file_version = Column(Integer, default=1, comment='文件版本')
    file_mtime = Column(Float, default=0.0, comment="文件修改时间")
    file_size = Column(Integer, default=0, comment="文件大小")
    create_time = Column(DateTime, default=func.now(), comment='创建时间')

    def __repr__(self):
        return f"<KnowledgeFile(id='{self.id}', file_name='{self.file_name}', file_ext='{self.file_ext}', kb_name='{self.kb_name}', document_loader_name='{self.document_loader_name}', text_splitter_name='{self.text_splitter_name}', file_version='{self.file_version}', create_time='{self.create_time}')>"


class KnowledgeBaseModel(Base):
    """
    知识库模型
    """
    __tablename__ = 'knowledge_base'
    id = Column(Integer, primary_key=True, autoincrement=True, comment='知识库ID')
    kb_name = Column(String(50), comment='知识库名称')
    kb_info = Column(String(200), comment='知识库简介 ')
    vs_type = Column(String(50), comment='向量库类型')
    embed_model = Column(String(50), comment='嵌入模型名称')
    file_count = Column(Integer, default=0, comment='文件数量')
    create_time = Column(DateTime, default=func.now(), comment='创建时间')

    def __repr__(self):
        return f"<KnowledgeBase(id='{self.id}', kb_name='{self.kb_name}',kb_intro='{self.kb_info} vs_type='{self.vs_type}', embed_model='{self.embed_model}', file_count='{self.file_count}', create_time='{self.create_time}')>"


class Prompt(Base):
    """
    知识库模型
    """
    __tablename__ = 'prompt'
    id = Column(Integer, primary_key=True,
                autoincrement=True, comment='prompt ID')
    name = Column(String(50), comment='prompt template name')
    template = Column(String(200), comment='prompt template content ')
    create_time = Column(DateTime, default=func.now(), comment='创建时间')

    def __repr__(self):
        return f"<Prompt(id='{self.id}', name='{self.name}',template='{self.template}  ')>"


Base.metadata.create_all(bind=engine)

session_factory = sessionmaker(bind=engine)


# session2 = sessionmaker(bind=engine)()

# @contextmanager
# # 上下文管理器用于管理Session的创建和关闭
# def session_scope():
#     """Provide a transactional scope around a series of operations."""
#     Session = scoped_session(session_factory)

#     session = Session()
#     # if session.is_closed():
#     #  logger.info("Session is closed.")
#     # else:
#     #     logger.info("Session is open.")

#     try:
#         yield session
#         # session.commit()
#     except:
#         # session.rollback()
#         raise
#     finally:
#         pass
#         # session.remove()

# def with_session(f):
#     @wraps(f)
#     def wrapper(*args, **kwargs):
#         with session_scope() as session:
#             try:
#                 result = f(session, *args, **kwargs)
#                 session.commit()
#                 return result
#             except:
#                 session.rollback()
#                 raise


#     return wrapper

def with_session(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            session2 = session_factory()
            result = f(session2, *args, **kwargs)
            session2.commit()
            return result
        except Exception as e:
            logger.info(str(e))
            session2.rollback()
            raise
        finally:
            session2.close()

    return wrapper


thread_pool = ThreadPoolExecutor(1)


def run_in_thread_pool(
        func: Callable,
        params: List[Dict] = [],
        pool: ThreadPoolExecutor = None,
) -> Generator:
    '''
    在线程池中批量运行任务，并将运行结果以生成器的形式返回。
    请确保任务中的所有操作是线程安全的，任务函数请全部使用关键字参数。
    '''
    tasks = []
    pool = pool or thread_pool

    for kwargs in params:
        thread = pool.submit(func, **kwargs)
        tasks.append(thread)

    for obj in as_completed(tasks):
        yield obj.result()


def _save_files_in_thread(files: List[UploadFile],
                          knowledge_base_name: str,
                          override: bool,
                          batch_prefix='./'):
    '''
    通过多线程将上传的文件保存到对应知识库目录内。
    生成器返回保存结果：{"code":200, "msg": "xxx", "data": {"knowledge_base_name":"xxx", "file_name": "xxx"}}
    '''

    def save_file(file: UploadFile, knowledge_base_name: str, override: bool) -> dict:
        '''
        保存单个文件。
        '''
        try:
            filename = file.filename
            file_content = file.file.read()  # 读取上传文件的内容

            # file_path = get_file_path(knowledge_base_name=knowledge_base_name, doc_name=filename)
            data = {"knowledge_base_name": knowledge_base_name,
                    "file_name": filename}
            contentRoot = get_content_root(knowledge_base_name)
            pathWithBatchPrefix = Path(batch_prefix) / filename
            fileDiskPath = Path(contentRoot) / pathWithBatchPrefix
            file_path = str(fileDiskPath)
            if not fileDiskPath.parent.exists():
                fileDiskPath.parent.mkdir(parents=True)
            if override:
                # vector_store = get_vs_from_kb(knowledge_base_name)
                delete_docs(knowledge_base_name, [str(pathWithBatchPrefix)])
                # for k,v in vector_store.docstore._dict.items():
                #     logger.info(1111,v.metadata.get("source"))
                # ids = [k for k, v in vector_store.docstore._dict.items() if v.metadata.get("source") == file_path]
                # if len(ids) > 0:
                #     vector_store.delete(ids)
                #     vector_store.save_local(vector_store.vs_path, 'index')
            if (os.path.isfile(file_path)
                    and not override
                    and os.path.getsize(file_path) == len(file_content)
            ):
                file_status = f"file {filename} existed in this batch"
                return dict(code=404, msg=file_status, data=data)

            file_path = os.path.expanduser(file_path)
            with open(file_path, "wb") as f:
                f.write(file_content)
            return dict(code=200, msg=f"成功上传文件 {filename}", data=data)
        except Exception as e:
            msg = f"{filename} 文件上传失败，报错信息为: {e}"
            return dict(code=500, msg=msg, data=data)

    params = [{"file": file, "knowledge_base_name": knowledge_base_name,
               "override": override} for file in files]
    for result in run_in_thread_pool(save_file, params=params):
        yield result


@with_session
def load_kb_from_db(session, kb_name):
    kb = session.query(KnowledgeBaseModel).filter_by(kb_name=kb_name).first()
    if kb:
        return copy.copy(kb)
    else:
        return None


@with_session
def queryEmbedSettingByFile(session, filepath):
    knowledgeFileModel: KnowledgeFileModel = (session.query(KnowledgeFileModel)
                                              .filter_by(filepath=filepath)
                                              .first())
    if knowledgeFileModel:
        return knowledgeFileModel.kb_name, knowledgeFileModel.chunk_size, knowledgeFileModel.chunk_overlap,
    else:
        return None, None, None


@with_session
def queryFilesByKB(session, KB):
    allfiles = (session.query(KnowledgeFileModel)
                .filter_by(kb_name=KB)
                .all())
    resultArr = []
    for kfile in allfiles:
        result = {
            'batch_name': kfile.batch_name,
            'file_basename': str(Path(kfile.file_name).relative_to(Path(kfile.batch_name))),
            'chunk_size': kfile.chunk_size,
            'chunk_overlap': kfile.chunk_overlap,
            'status': kfile.status,
            'msg': kfile.msg,

        }
        resultArr.append(result)

    # return  json.dumps(resultArr, ensure_ascii=False)
    return resultArr


@with_session
def add_batchInfo_to_db(session, batchInfo: KnowledgeBatchInfo):
    batchInfoDB = session.query(KnowledgeBatchInfo).filter_by(batch_name=batchInfo.batch_name,
                                                              kb_name=batchInfo.kb_name).first()
    if batchInfoDB:
        batchInfoDB.update_time = datetime.now()
    else:
        session.add(batchInfo)


@with_session
def delete_batchInfo_in_db(session, batchInfo: KnowledgeBatchInfo):
    batchInfoDB = session.query(KnowledgeBatchInfo).filter_by(batch_name=batchInfo.batch_name,
                                                              kb_name=batchInfo.kb_name).first()
    if batchInfoDB:
        session.delete(batchInfoDB)
        return True
    else:
        return True


@with_session
def delete_allbatchInfo_in_one_kb(session, batchInfo: KnowledgeBatchInfo):
    batchInfoDB = session.query(KnowledgeBatchInfo).filter_by(
        kb_name=batchInfo.kb_name).delete()
    return True


@with_session
def add_file_to_db(session,
                   kb_file: KnowledgeFile,
                   ):
    kb = session.query(KnowledgeBaseModel).filter_by(
        kb_name=kb_file.knowledge_base_name).first()
    if kb:
        # 如果已经存在该文件，则更新文件信息与版本号
        existing_file: KnowledgeFileModel = (session.query(KnowledgeFileModel)
                                             .filter_by(filepath=kb_file.filepath)
                                             .first())
        mtime = kb_file.get_mtime()
        size = kb_file.get_size()

        if existing_file:
            existing_file.file_mtime = mtime
            existing_file.file_size = size
            existing_file.file_version += 1
        # 否则，添加新文件
        else:
            new_file = KnowledgeFileModel(
                file_name=kb_file.filename,
                file_ext=kb_file.ext,
                filepath=kb_file.filepath,
                batch_name=kb_file.batch_name,
                status=kb_file.status,
                msg=kb_file.msg,
                chunk_overlap=kb_file.chunk_overlap,
                chunk_size=kb_file.chunk_size,
                kb_name=kb_file.knowledge_base_name,
                document_loader_name=kb_file.document_loader_name,
                text_splitter_name=kb_file.text_splitter_name or "SpacyTextSplitter",
                file_mtime=mtime,
                file_size=size,
            )
            kb.file_count += 1
            session.add(new_file)
    return True


@with_session
def add_kb_to_db(session, kb_name, kb_info, vs_type, embed_model):
    # 创建知识库实例
    kb = session.query(KnowledgeBaseModel).filter_by(kb_name=kb_name).first()
    if not kb:
        kb = KnowledgeBaseModel(
            kb_name=kb_name, kb_info=kb_info, vs_type=vs_type, embed_model=embed_model)
        session.add(kb)
    else:  # update kb with new vs_type and embed_model
        kb.kb_info = kb_info
        kb.vs_type = vs_type
        kb.embed_model = embed_model
    return True


@with_session
def list_kbs_from_db(session, min_file_count: int = -1):
    kbs = session.query(KnowledgeBaseModel.kb_name).filter(
        KnowledgeBaseModel.file_count > min_file_count).all()
    kbs = [kb[0] for kb in kbs]
    return kbs


def list_kbs():
    # Get List of Knowledge Base
    return ListResponse(data=list_kbs_from_db())


def list_vector_store_types():
    return ListResponse(data=list(config.VECTOR_STORE_DICT))


def list_embedding_models():
    return ListResponse(data=list(config.EMBEDDING_DICT.keys()))


def text_embedding(
        text: str = Body(..., description="prompt name",
                         examples=["you are cool"]),
        embed_model: str = Body(...,
                                description="when you chat with llm, {query} is variable, chat with rag, {query} {context} are variables",
                                examples=["bge_m3"]),
) -> BaseResponse:
    print("##text:", text)
    print("##embed_model:", embed_model)
    embeddingModel = config.EMBEDDING_DICT.get(embed_model)
    query_vector = embeddingModel.embed_query(text)

    return BaseResponse(data=str(query_vector))


def list_llms():
    # Get List of Knowledge Base

    return ListResponse(data=list(config.MODEL_DICT.keys()))


def get_llm_info(llm: str = Query(..., description="llm name", examples=["ociGenAI_command"])):
    return BaseResponse(data=config.MODEL_DICT.get(llm))


def checkIfKBExists(kb_name: str
                    ):
    kb = load_kb_from_db(kb_name)
    # if vs_type is None and os.path.isdir(get_kb_path(kb_name)):  # faiss knowledge base not in db
    # vs_type = "faiss"
    return kb


def create_dir(kbName):
    kbpath = get_kb_path(kbName)
    docPath = get_content_root(kbName)
    url_subpath = get_url_subpath(kbName)
    upload_subpath = get_uploaded_file_subpath(kbName)
    vspath = get_vs_path(kbName)
    if not os.path.exists(kbpath):
        os.makedirs(kbpath)
    if not os.path.exists(docPath):
        os.makedirs(docPath)
    if not os.path.exists(vspath):
        os.makedirs(vspath)
    if not os.path.exists(url_subpath):
        os.makedirs(url_subpath)
    # if not os.path.exists(upload_subpath):
    #     os.makedirs(upload_subpath)


# especially for faiss


def init_vs(knowledge_base_name, embed_model, vector_store_type):
    '''
    embed_model is a string, should be converted to model
    '''
    vs_path = get_vs_path(knowledge_base_name)
    embeddingModel = config.EMBEDDING_DICT.get(embed_model)
    vector_store = 1
    if vector_store_type == 'faiss':
        if not os.path.exists(vs_path):
            os.makedirs(vs_path)
        if os.path.isfile(os.path.join(vs_path, "index.faiss")):
            vector_store = FAISS.load_local(
                vs_path, embeddingModel, normalize_L2=True)
        else:
            # create an empty vector store
            doc = Document(page_content="init", metadata={})
            vector_store = FAISS.from_documents(
                [doc], embeddingModel, normalize_L2=True)
            ids = list(vector_store.docstore._dict.keys())
            vector_store.delete(ids)
            vector_store.save_local(vs_path)
            vector_store.vs_path = vs_path

    elif vector_store_type == 'opensearch':
        vector_store = OpenSearchVectorSearch(
            index_name=knowledge_base_name,
            embedding_function=embeddingModel,
            opensearch_url=config.OCI_OPEN_SEARCH_URL,
            http_auth=(config.OCI_OPEN_SEARCH_USER,
                       config.OCI_OPEN_SEARCH_PASSWD)
        )
    elif vector_store_type == 'oracle':
        # vector_store = OracleAIVector(collection_name=knowledge_base_name,
        #                              connection_string=config.ORACLE_AI_VECTOR_CONNECTION_STRING,
        #                              pre_delete_collection=False,
        #                              embedding_function=embeddingModel
        #                              )
        # vector_store.create_tables_if_not_exists()
        vector_store = OracleBaseDbVS(collection_name=knowledge_base_name,
                                      # connection_string=config.ORACLE_AI_VECTOR_CONNECTION_STRING,
                                      pre_delete_collection=False,
                                      embedding_function=embeddingModel
                                      )
    elif vector_store_type == 'heatwave':
        vector_store = HeatWaveVS(
            collection_name=knowledge_base_name,
            pre_delete_collection=False,
            embedding_function=embeddingModel
        )
    elif vector_store_type == 'adb':
        vector_store = OracleAdbVS(
            collection_name=knowledge_base_name,
            pre_delete_collection=False,
            embedding_function=embeddingModel
        )

    return vector_store


def get_vs_from_kb(kb_name):
    kb = load_kb_from_db(kb_name)
    if 'faiss' == kb.vs_type:
        vs_path = get_vs_path(kb_name)
        vector_store = FAISS.load_local(vs_path, config.EMBEDDING_DICT[kb.embed_model], normalize_L2=True,
                                        allow_dangerous_deserialization=True)
        vector_store.vs_path = vs_path
        return vector_store, kb
    elif 'oracle' == kb.vs_type:
        # vector_store = OracleAIVector.from_existing_index(
        #    embedding=config.EMBEDDING_DICT[kb.embed_model],
        #    collection_name=kb_name,
        #    embedding_function=config.EMBEDDING_DICT[kb.embed_model],
        # )
        vector_store = OracleBaseDbVS(
            collection_name=kb_name,
            embedding_function=config.EMBEDDING_DICT[kb.embed_model],
        )
        return vector_store, kb
    elif 'opensearch' == kb.vs_type:
        vector_store = OpenSearchVectorSearch(
            index_name=kb_name,
            embedding_function=config.EMBEDDING_DICT[kb.embed_model],
            opensearch_url=config.OCI_OPEN_SEARCH_URL,
            http_auth=(config.OCI_OPEN_SEARCH_USER,
                       config.OCI_OPEN_SEARCH_PASSWD)
        )
    elif 'heatwave' == kb.vs_type:
        vector_store = HeatWaveVS(
            collection_name=kb_name,
            embedding_function=config.EMBEDDING_DICT[kb.embed_model],
        )
        return vector_store, kb
    elif 'adb' == kb.vs_type:
        vector_store = OracleAdbVS(
            collection_name=kb_name,
            embedding_function=config.EMBEDDING_DICT[kb.embed_model],
        )
        return vector_store, kb


class UploadFromUrlRequest(pydantic.BaseModel):
    urls: List[str]
    knowledge_base_name: str
    batch_name: str = './'
    max_depth: int = pydantic.Field(1, description="max_depth  ")
    chunk_size: int = pydantic.Field(
        config.CHUNK_SIZE, description="Response text")
    chunk_overlap: int = pydantic.Field(
        config.CHUNK_OVERLAP, description="Response text")

    class Config:
        json_schema_extra = {
            "knowledge_base_name": "d",
            "urls": ["www.qq.com", "www.baidu.com"],
            "batch_name": "./",
            "max_depth": 1,
            "chunk_size": 123,
            "chunk_overlap": 12

        }


def upload_from_url(upload_url_request: UploadFromUrlRequest):
    failed_files = {}
    URLS = []
    kb = checkIfKBExists(upload_url_request.knowledge_base_name)
    for url in upload_url_request.urls:
        kb_file = None
        try:

            loader = RecursiveUrlLoader(
                url,
                max_depth=upload_url_request.max_depth,
            )
            docs = loader.load()
            for doc in docs:
                url = doc.metadata["source"]
                URLS.append(url)

            for childUrl in URLS:
                # doc.page_content[:]
                # childUrl=doc.metadata['source']
                kb_file = None
                if not upload_url_request.batch_name.endswith('/'):
                    kb_file = KnowledgeFile(filepath=childUrl, type='webpage', ext='.url',
                                            batch_name=upload_url_request.batch_name,
                                            filename=upload_url_request.batch_name + '/' + childUrl,
                                            knowledge_base_name=upload_url_request.knowledge_base_name,
                                            chunk_overlap=upload_url_request.chunk_overlap,
                                            chunk_size=upload_url_request.chunk_size)
                else:
                    kb_file = KnowledgeFile(filepath=childUrl, type='webpage', ext='.url',
                                            batch_name=upload_url_request.batch_name,
                                            filename=upload_url_request.batch_name + childUrl,
                                            knowledge_base_name=upload_url_request.knowledge_base_name,
                                            chunk_overlap=upload_url_request.chunk_overlap,
                                            chunk_size=upload_url_request.chunk_size)
                batchInfo = KnowledgeBatchInfo(batch_name=upload_url_request.batch_name,
                                               kb_name=upload_url_request.knowledge_base_name,
                                               chunk_size=upload_url_request.batch_name,
                                               chunk_overlap=upload_url_request.chunk_overlap)
                Text_splitter = makeSplitter(
                    upload_url_request.chunk_size, upload_url_request.chunk_overlap)
                chunkDocuments = kb_file.url2Docs(Text_splitter)
                delete_webpage(
                    upload_url_request.knowledge_base_name, str(childUrl))

                store_vectors_by_batch(
                    upload_url_request.knowledge_base_name, kb.embed_model, chunkDocuments)

                add_file_to_db(kb_file)
                add_batchInfo_to_db(batchInfo)
        except Exception as e:
            msg = f"embedding ‘{kb_file.filename}’ to ‘{upload_url_request.knowledge_base_name}’ failed with some error {e}"
            failed_files[kb_file.filename] = msg
            kb_file.msg = msg
            kb_file.status = 'failure'
            logger.error(e)
            # add_file_to_db(kb_file)
            # add_batchInfo_to_db(batchInfo)
    return BaseResponse(code=200, msg="url uploading completed", data={"failed_files": failed_files})


@with_session
def delete_file_from_db(session, kb_file: KnowledgeFile):
    existing_file = session.query(KnowledgeFileModel).filter_by(
        filepath=kb_file.filepath).first()
    if existing_file:
        session.delete(existing_file)

        kb = session.query(KnowledgeBaseModel).filter_by(
            kb_name=kb_file.knowledge_base_name).first()
        if kb:
            kb.file_count -= 1
        #     batchExisted = session.query(BatchInfo).filter_by(batchname=kb_file.batch_name, kb=kb_file.knowledge_base_name).first()
        #     batchExisted.update_time = datetime.now()
        session.commit()
    return True


def delete_batch(knowledge_base_name: str = Body(..., examples=["samples"]),
                 batch_name: str = Body(..., examples=["images"]),
                 ):
    docRootPath = get_content_root(knowledge_base_name)
    batchRoot = Path(docRootPath) / Path(batch_name)

    #
    # all_files = [file for file in docRootPath.rglob('*') if file.is_file()
    #              and file.suffix.lower() not in LOADER_DICT['ImageOCRLoader']
    #              and file.suffix.lower() not in LOADER_DICT['OCISpeechLoader']]
    # filter the files not begin with ocr and asr
    files_with_batch_name = [str(Path(file).relative_to(docRootPath)) for file in batchRoot.rglob('*') if
                             file.is_file()]
    resp = delete_docs(knowledge_base_name, filenames=files_with_batch_name)
    os.rmdir(str(batchRoot))
    batchInfo = KnowledgeBatchInfo(batch_name=batch_name,
                                   kb_name=knowledge_base_name,
                                   )
    delete_batchInfo_in_db(batchInfo)
    return resp


def delete_webpage(knowledge_base_name: str = Body(..., examples=["samples"]),
                   url=Body(..., examples=["https://xxxx"])) -> DeleteResponse:
    knowledge_base_name = urllib.parse.unquote(knowledge_base_name)
    type = 'webpage'
    kb = checkIfKBExists(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")
    failed_files = {}
    # contentRoot = get_content_root(knowledge_base_name)

    kb_file = KnowledgeFile(filepath=url,
                            knowledge_base_name=knowledge_base_name, type=type)
    vector_store, _ = get_vs_from_kb(knowledge_base_name)
    logger.info("## deleting file: {}", kb_file.filepath)
    # check if the file exists
    status = delete_file_from_db(kb_file)
    if status == False:
        failed_files[filename] = 'Failed to delete it in sqlite db'

    if (
            isinstance(vector_store, OracleAIVector)
            or isinstance(vector_store, HeatWaveVS)
            or isinstance(vector_store, OracleAdbVS)
            or isinstance(vector_store, OracleBaseDbVS)
    ):
        try:
            vector_store.delete_embeddings([kb_file.filepath])
        except Exception as e:
            logger.error(e)
            failed_files[filename] = 'Failed to delete it in vectorDB'
            pass
    else:
        ids = [k for k, v in vector_store.docstore._dict.items() if
               v.metadata.get("source") == kb_file.filepath]
        #    check faiss content
        # logger.info(vector_store.docstore._dict)
        if len(ids) > 0:
            vector_store.delete(ids)
            vector_store.save_local(vector_store.vs_path, 'index')

    return DeleteResponse(code=200, msg=f"webpage file deletion ended", data={"failed_files": failed_files})


def delete_docs(knowledge_base_name: str = Body(..., examples=["samples"]),
                filenames: List[str] = Body(..., examples=[
                    ["file_name.md", "test.txt"]]),
                ) -> DeleteResponse:
    knowledge_base_name = urllib.parse.unquote(knowledge_base_name)
    kb = checkIfKBExists(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")
    failed_files = {}
    # contentRoot = get_content_root(knowledge_base_name)

    for filename in filenames:
        kb_file = KnowledgeFile(filename=filename,
                                knowledge_base_name=knowledge_base_name)
        vector_store, _ = get_vs_from_kb(knowledge_base_name)
        logger.info("## deleting file: {}", kb_file.filepath)
        # check if the file exists
        if os.path.exists(kb_file.filepath):
            # python delete a file from os
            os.remove(kb_file.filepath)
            status = delete_file_from_db(kb_file)
            if status == False:
                failed_files[filename] = 'Failed to delete it in sqlite db'

            if (
                    isinstance(vector_store, OracleAIVector)
                    or isinstance(vector_store, HeatWaveVS)
                    or isinstance(vector_store, OracleAdbVS)
                    or isinstance(vector_store, OracleBaseDbVS)
            ):
                try:
                    vector_store.delete_embeddings([kb_file.filepath])
                except Exception as e:
                    logger.error(e)
                    failed_files[filename] = 'Failed to delete it in vectorDB'
                    pass
            else:
                ids = [k for k, v in vector_store.docstore._dict.items() if
                       v.metadata.get("source") == kb_file.filepath]
                #    check faiss content
                # logger.info(vector_store.docstore._dict)
                if len(ids) > 0:
                    vector_store.delete(ids)
                    vector_store.save_local(vector_store.vs_path, 'index')
        else:
            failed_files[filename] = 'File does not exist'

    return DeleteResponse(code=200, msg=f"File deletion ended", data={"failed_files": failed_files})


def saveInVectorDB(knowledge_base_name, texts: List[Document]):
    '''

    :param knowledge_base_name:
    :param texts:  the chunk documents
    :return:
    '''
    kb = load_kb_from_db(knowledge_base_name)
    if 'faiss' == kb.vs_type:
        vs_path = get_vs_path(knowledge_base_name)
        vector_store = FAISS.load_local(vs_path, config.EMBEDDING_DICT[kb.embed_model], normalize_L2=True,
                                        allow_dangerous_deserialization=True)
        vector_store.add_documents(texts)
        vector_store.save_local(vs_path, 'index')
    elif 'oracle' == kb.vs_type:
        # vector_store = OracleAIVector.from_documents(
        #    embedding=config.EMBEDDING_DICT[kb.embed_model],
        #    documents=texts,
        #    collection_name=knowledge_base_name,
        #    connection_string=config.ORACLE_AI_VECTOR_CONNECTION_STRING,
        #    pre_delete_collection=False,  # Append to the vectorstore
        # )
        vector_store = OracleBaseDbVS.from_documents(
            embedding=config.EMBEDDING_DICT[kb.embed_model],
            documents=texts,
            collection_name=knowledge_base_name,
            connection_string=config.ORACLE_AI_VECTOR_CONNECTION_STRING,
            pre_delete_collection=False,  # Append to the vectorstore
        )
    elif 'opensearch' == kb.vs_type:
        vector_store = OpenSearchVectorSearch.from_documents(
            embedding=config.EMBEDDING_DICT[kb.embed_model],
            documents=texts,
            index_name=knowledge_base_name,
            opensearch_url=config.OCI_OPEN_SEARCH_URL,
            http_auth=(config.OCI_OPEN_SEARCH_USER,
                       config.OCI_OPEN_SEARCH_PASSWD),
            bulk_size=2222,
        )
    elif 'heatwave' == kb.vs_type:
        vector_store = HeatWaveVS.from_documents(
            collection_name=knowledge_base_name,
            documents=texts,
            embedding=config.EMBEDDING_DICT[kb.embed_model],
            pre_delete_collection=False,  # Append to the vectorstore
        )
    elif 'adb' == kb.vs_type:
        vector_store = OracleAdbVS.from_documents(
            collection_name=knowledge_base_name,
            documents=texts,
            embedding=config.EMBEDDING_DICT[kb.embed_model],
            pre_delete_collection=False,  # Append to the vectorstore
        )

    return vector_store


def queryInVectorDB(knowledge_base_name, texts: List[Document]):
    '''

    :param knowledge_base_name:
    :param texts:  the chunk documents
    :return:
    '''
    kb = load_kb_from_db(knowledge_base_name)
    if 'faiss' == kb.vs_type:
        vs_path = get_vs_path(knowledge_base_name)
        vector_store = FAISS.load_local(vs_path, config.EMBEDDING_DICT[kb.embed_model], normalize_L2=True,
                                        allow_dangerous_deserialization=True)
        vector_store.add_documents(texts)
        vector_store.save_local(vs_path, 'index')
    elif 'oracle' == kb.vs_type:
        # vector_store = OracleAIVector.from_documents(
        #    embedding=config.EMBEDDING_DICT[kb.embed_model],
        #    documents=texts,
        #    collection_name=knowledge_base_name,
        #    connection_string=config.ORACLE_AI_VECTOR_CONNECTION_STRING,
        #    pre_delete_collection=False,  # Append to the vectorstore
        # )
        vector_store = OracleBaseDbVS.from_documents(
            collection_name=knowledge_base_name,
            documents=texts,
            embedding=config.EMBEDDING_DICT[kb.embed_model],
            pre_delete_collection=False,  # Append to the vectorstore
        )
    elif 'opensearch' == kb.vs_type:
        vector_store = OracleAIVector.from_documents(
            embedding=config.EMBEDDING_DICT[kb.embed_model],
            documents=texts,
            collection_name=knowledge_base_name,
            connection_string=config.ORACLE_AI_VECTOR_CONNECTION_STRING,
            pre_delete_collection=False,  # Append to the vectorstore
        )
    elif 'heatwave' == kb.vs_type:
        vector_store = HeatWaveVS.from_documents(
            collection_name=knowledge_base_name,
            documents=texts,
            embedding=config.EMBEDDING_DICT[kb.embed_model],
            pre_delete_collection=False,  # Append to the vectorstore
        )
    elif 'adb' == kb.vs_type:
        vector_store = OracleAdbVS.from_documents(
            collection_name=knowledge_base_name,
            documents=texts,
            embedding=config.EMBEDDING_DICT[kb.embed_model],
            pre_delete_collection=False,  # Append to the vectorstore
        )

    return vector_store


executor = concurrent.futures.ThreadPoolExecutor(max_workers=99)


def upload_audio_from_object_storage(
        namespace: str = Body(..., description="bucket namespace", examples=[
            "sehubjapacprod"]),
        bucket: str = Body('testo', description="bucket name"),
        object_prefix: str = Body(
            '', description="object_prefix e.g. folder1/"),
        batch_name: str = Body('./',
                               description="knowledge base batch prefix name for this batch uploading  e.g. batch1"),
        knowledge_base_name: str = Body(..., description="knowledge_base_name  ", examples=[
            "samples"]),
        chunk_size: int = Body(config.CHUNK_SIZE, description="知识库中单段文本最大长度"),
        chunk_overlap: int = Body(
            config.CHUNK_OVERLAP, description="知识库中相邻文本重合长度"),
        language: str = Body('en', description="language for ASR this audio"),
) -> BaseResponse:
    if knowledge_base_name is None or knowledge_base_name.strip() == "":
        return BaseResponse(code=404, msg="kb name can not be empty ")
    kb = checkIfKBExists(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"not found this kb {knowledge_base_name}")
    content_root = get_content_root(knowledge_base_name)
    # for future support different regions
    # config_oci = {'region': signer.region, 'tenancy':  signer.tenancy_id}
    futures = []
    object_storage_client = oci.object_storage.ObjectStorageClient(
        **init_oci_auth(config.auth_type))

    def output():
        next_starts_with = None
        saved_disk_files = []
        while True:
            try:
                response = object_storage_client.list_objects(namespace, bucket, start=next_starts_with,
                                                              prefix=object_prefix,
                                                              fields='size,timeCreated,timeModified,storageTier',
                                                              retry_strategy=oci.retry.DEFAULT_RETRY_STRATEGY)

            except Exception as e:
                progress_obj = {
                    'errors': str(e),
                }
                check_vdb_init_status[knowledge_base_name] = progress_obj
                logger.info('errors ', str(e))
                return BaseResponse(code=500, msg=str(e))

            next_starts_with = response.data.next_start_with
            for object_file in response.data.objects:

                obj_resp = object_storage_client.get_object(
                    namespace_name=namespace,
                    bucket_name=bucket,
                    object_name=object_file.name,
                )
                data = obj_resp.data
                objectNameWithBatchPrefix = Path(
                    batch_name) / Path(object_file.name)

                if object_file.name.endswith('.wav'):
                    # delete old vs data and sqlite data and os fs data
                    delete_docs(knowledge_base_name, [
                        str(objectNameWithBatchPrefix)])
                    logger.info(f'audio file: {object_file.name}')
                    futures.append(
                        executor.submit(write_object, content_root, str(objectNameWithBatchPrefix), data))
                    progress_obj = {
                        'details': f'downloading {object_file.name} ',
                    }
                    check_vdb_init_status[knowledge_base_name] = progress_obj
                    yield f'downloading  {object_file.name} from bucket \n'
                    saved_disk_files.append(str(objectNameWithBatchPrefix))

            if not next_starts_with:
                break
        concurrent.futures.wait(futures)
        kb_files = [KnowledgeFile(type='audio', knowledge_base_name=knowledge_base_name,
                                  filename=objectNameWithBatchPrefix) for objectNameWithBatchPrefix in
                    saved_disk_files]

        logger.info('vectorize audio files .....')
        i = 0
        msg_array = []
        msg = ""
        for kb_file in tqdm(kb_files):
            try:
                kb_file.batch_name = batch_name
                kb_file.chunk_overlap = chunk_overlap
                kb_file.chunk_size = chunk_size
                kb_file.knowledge_base_name = knowledge_base_name
                batchInfo = KnowledgeBatchInfo(batch_name=batch_name,
                                               kb_name=knowledge_base_name,
                                               chunk_size=chunk_size,
                                               chunk_overlap=chunk_overlap)
                text_splitter = makeSplitter(chunk_size, chunk_overlap)
                audioAbsolutePath = kb_file.filepath
                parentDir = Path(content_root) / Path(batch_name)
                objectNameInBucket = Path(audioAbsolutePath).relative_to(parentDir)

                # batchPath =Path()
                # relative_paths = [str(relativePath ) for file in all_files]

                texts = ociSpeechASRLoader(
                    namespace, bucket, str(objectNameInBucket), language)
                kb_file.full_text = texts
                chunkDocuments = kb_file.file2text(text_splitter)
                store_vectors_by_batch(
                    knowledge_base_name, kb.embed_model, chunkDocuments)

            except Exception as e:
                msg = f"添加文件‘{kb_file.filename}’到vector store‘{knowledge_base_name}’时vectorize出错： {e}"
                kb_file.msg = msg
                kb_file.status = 'failure'
                logger.error(msg)
            i = i + 1
            progress_obj = {
                "total": len(kb_files),
                "finished": i,
                "current_document": kb_file.filename,
                'details': msg_array,
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap
            }
            progress_obj['details'].append(
                {'file_name': kb_file.filename, 'msg': msg})

            check_vdb_init_status[knowledge_base_name] = progress_obj
            add_file_to_db(kb_file)
            add_batchInfo_to_db(batchInfo)

            yield f'embedding {kb_file.filename} \n'

    return StreamingResponse(output(), media_type="text/event-stream")


async def upload_from_object_storage(
        namespace: str = Body(..., description="bucket namespace", examples=[
            "sehubjapacprod"]),
        bucket: str = Body('testo', description="bucket name"),
        object_prefix: str = Body(
            '', description="object_prefix e.g. folder1/"),
        batch_name: str = Body('./',
                               description="knowledge base batch prefix name for this batch uploading  e.g. batch1"),
        knowledge_base_name: str = Body(..., description="knowledge_base_name  ", examples=[
            "samples"]),
        chunk_size: int = Body(config.CHUNK_SIZE, description="知识库中单段文本最大长度"),
        chunk_overlap: int = Body(
            config.CHUNK_OVERLAP, description="知识库中相邻文本重合长度"),
) -> BaseResponse:
    if knowledge_base_name is None or knowledge_base_name.strip() == "":
        return BaseResponse(code=404, msg="知识库名称不能为空，请重新填写知识库名称")
    kb = checkIfKBExists(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")
    content_root = get_content_root(knowledge_base_name)
    # for future support different regions
    # config_oci = {'region': signer.region, 'tenancy':  signer.tenancy_id}
    futures = []

    object_storage_client = oci.object_storage.ObjectStorageClient(
        **init_oci_auth(config.auth_type))

    def output():
        next_starts_with = None
        saved_disk_files = []
        while True:
            try:
                response = object_storage_client.list_objects(namespace, bucket, start=next_starts_with,
                                                              prefix=object_prefix,
                                                              fields='size,timeCreated,timeModified,storageTier',
                                                              retry_strategy=oci.retry.DEFAULT_RETRY_STRATEGY)

            except Exception as e:
                progress_obj = {
                    'errors': str(e),
                }
                check_vdb_init_status[knowledge_base_name] = progress_obj
                logger.info('errors {}', str(e))
                return BaseResponse(code=500, msg=str(e))

            next_starts_with = response.data.next_start_with
            for object_file in response.data.objects:
                logger.info(f'object in bucket : {object_file.name}')
                obj_resp = object_storage_client.get_object(
                    namespace_name=namespace,
                    bucket_name=bucket,
                    object_name=object_file.name,
                )
                data = obj_resp.data
                # delete old vs data and sqlite data and os fs data
                delete_docs(knowledge_base_name, [object_file.name])
                if data.content == b'':
                    continue
                parentDir = Path(content_root) / Path(batch_name)
                futures.append(executor.submit(
                    write_object, str(parentDir), object_file.name, data))
                progress_obj = {
                    'details': f'downloading {object_file.name} ',
                }
                check_vdb_init_status[knowledge_base_name] = progress_obj
                yield f'downloading {object_file.name} from bucket \n'
                fileNameInKB = Path(batch_name) / Path(object_file.name)
                saved_disk_files.append(str(fileNameInKB))

            if not next_starts_with:
                break
        concurrent.futures.wait(futures)
        kb_files = [KnowledgeFile(
            filename=file, knowledge_base_name=knowledge_base_name) for file in saved_disk_files]

        logger.info('vectorize.....')
        i = 0
        msg_array = []
        msg = ""
        for kb_file in tqdm(kb_files):
            try:

                kb_file.batch_name = batch_name
                kb_file.chunk_overlap = chunk_overlap
                kb_file.chunk_size = chunk_size
                kb_file.knowledge_base_name = knowledge_base_name
                batchInfo = KnowledgeBatchInfo(batch_name=batch_name,
                                               kb_name=knowledge_base_name,
                                               chunk_size=chunk_size,
                                               chunk_overlap=chunk_overlap)

                Text_splitter = makeSplitter(chunk_size, chunk_overlap)
                chunkDocuments = kb_file.file2text(Text_splitter)

                store_vectors_by_batch(
                    knowledge_base_name, kb.embed_model, chunkDocuments)

            except Exception as e:
                msg = f"添加文件‘{kb_file.filename}’到vector store ‘{knowledge_base_name}’ 时vectorize出错：{e}"

                logger.error(msg)
                kb_file.msg = msg
                kb_file.status = 'failure'

            i = i + 1
            progress_obj = {
                "total": len(kb_files),
                "finished": i,
                "current_document": kb_file.filename,
                'details': msg_array,
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap
            }
            progress_obj['details'].append(
                {'file_name': kb_file.filename, 'msg': msg})

            check_vdb_init_status[knowledge_base_name] = progress_obj
            add_file_to_db(kb_file)
            add_batchInfo_to_db(batchInfo)
            yield f'embedding {kb_file.filename} \n'

    return StreamingResponse(output(), media_type="text/event-stream")


def upload_docs(files: List[UploadFile] = File(..., description="上传文件，支持多文件"),
                batch_name: str = Form("./", description="the prefix directory path for this batch file uploading",
                                       examples=["./"]),
                knowledge_base_name: str = Form(..., description="知识库名称", examples=[
                    "samples"]),
                override: bool = Form(False, description="覆盖已有文件"),
                chunk_size: int = Form(
                    config.CHUNK_SIZE, description="知识库中单段文本最大长度"),
                chunk_overlap: int = Form(
                    config.CHUNK_OVERLAP, description="知识库中相邻文本重合长度"),
                ocr_lang: str = Form('en',
                                     description="if having images, use this to define language in the image,eg. `ch`, `en`, `fr`, `german`, `korean`, `japan`")
                ) -> BaseResponse:
    '''
    API接口：上传文件，并/或向量化
    '''
    if knowledge_base_name is None or knowledge_base_name.strip() == "":
        return BaseResponse(code=404, msg="知识库名称不能为空，请重新填写知识库名称")
    kb = checkIfKBExists(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    failed_files = {}
    # file_names = list()

    # 先将上传的文件保存到磁盘
    saved_disk_files = []
    for result in _save_files_in_thread(files, knowledge_base_name=knowledge_base_name, override=override,
                                        batch_prefix=batch_name):
        filename = result["data"]["file_name"]
        if result["code"] != 200:
            failed_files[filename] = result["msg"]
        else:
            saved_disk_files.append(filename)

    kb_files = [KnowledgeFile(filename=str(Path(batch_name) / file), knowledge_base_name=knowledge_base_name) for file
                in saved_disk_files]
    logger.info('vectorize uploaded docs...')
    for kb_file in tqdm(kb_files):
        try:
            batchInfo = KnowledgeBatchInfo(batch_name=batch_name,
                                           kb_name=knowledge_base_name,
                                           chunk_size=chunk_size,
                                           chunk_overlap=chunk_overlap)
            text_splitter = makeSplitter(chunk_size, chunk_overlap)
            chunkDocuments = kb_file.file2text(text_splitter, ocr_lang)
            store_vectors_by_batch(knowledge_base_name,
                                   kb.embed_model, chunkDocuments)
            kb_file.batch_name = batch_name

            kb_file.chunk_overlap = chunk_overlap
            kb_file.chunk_size = chunk_size
            kb_file.knowledge_base_name = knowledge_base_name

            add_file_to_db(kb_file)
            add_batchInfo_to_db(batchInfo)

        except Exception as e:
            msg = f"embedding ‘{kb_file.filename}’ to ‘{knowledge_base_name}’ failed with some error {e}"
            failed_files[kb_file.filename] = msg
            kb_file.msg = msg
            kb_file.status = 'failure'
            logger.error(e)
            add_file_to_db(kb_file)
            add_batchInfo_to_db(batchInfo)

    return BaseResponse(code=200, msg="files uploaded and embedded", data={"failed_files": failed_files})


# 按批次插入embedding结果到向量数据库中。
def store_vectors_by_batch(knowledge_base_name, embed_model, chunkDocuments):
    if embed_model == "oci_genai_embed" or embed_model == "cohere_embed":
        total_chunks = 0
        if len(chunkDocuments) > 95:
            page_chunks = round(len(chunkDocuments) / 95)
        else:
            page_chunks = 1
        # logger.info(f"##page_chunks:{page_chunks}")
        # logic to handle OCIGenAIEmbeddings 95 docs to embed at a time
        for page_chunk in range(page_chunks):
            # filtered_docs = []
            selected_pages = chunkDocuments[page_chunk *
                                            95:(page_chunk + 1) * 95]
            total_chunks += len(selected_pages)
            if len(selected_pages) > 0:
                saveInVectorDB(knowledge_base_name, selected_pages)
        # logger.info("Total Number of chunks created ", total_chunks)
    else:
        saveInVectorDB(knowledge_base_name, chunkDocuments)


class VectorSearchResponse(pydantic.BaseModel):
    data: list = pydantic.Field(...,
                                description="data returned from vector store")
    status: str = pydantic.Field(..., description="Response text")
    err_msg: str = pydantic.Field(..., description="Response text")

    class Config:
        json_schema_extra = {
            "example": {
                "data": [{"content": "xxx", "score": 1, "source": "llm"},
                         {"content": "yyy", "score": 0.78, "source": "source file url"}],
                "status": "success",
                "err_msg": ""
            }
        }


@with_session
def delete_files_from_db(session, knowledge_base_name: str):
    session.query(KnowledgeFileModel).filter_by(
        kb_name=knowledge_base_name).delete()
    kb = session.query(KnowledgeBaseModel).filter_by(
        kb_name=knowledge_base_name).first()
    if kb:
        kb.file_count = 0

    session.commit()
    return True


@with_session
def delete_kb_from_db(session, knowledge_base_name: str):
    session.query(KnowledgeBaseModel).filter_by(
        kb_name=knowledge_base_name).delete()

    session.commit()
    return True


def list_files_from_folder(kb_name: str):
    docRootPath = get_content_root(kb_name)
    docRootPath = Path(docRootPath)
    #
    # all_files = [file for file in docRootPath.rglob('*') if file.is_file()
    #              and file.suffix.lower() not in LOADER_DICT['ImageOCRLoader']
    #              and file.suffix.lower() not in LOADER_DICT['OCISpeechLoader']]
    # filter the files not begin with ocr and asr
    all_real_files = [str(file) for file in docRootPath.rglob('*') if file.is_file()
                      and not str(file).startswith('ocr')
                      and not str(file).startswith('asr')]
    # Get relative paths to the root directory
    # relative_paths = [str(file.relative_to(docRootPath)) for file in all_files]

    return all_real_files


def files2docs_in_thread(
        files: List[Union[KnowledgeFile, Tuple[str, str], Dict]],
        splitter: TextSplitter,
        pool: ThreadPoolExecutor
) -> Generator:
    '''
    利用多线程批量将磁盘文件转化成langchain Document.
    生成器返回值为 status, (kb_name, file_name, docs | error)
    '''

    def file2docs(*, file: KnowledgeFile, **kwargs) -> Tuple[bool, Tuple[str, str, List[Document]]]:
        try:
            return True, (file.kb_name, file.filename, file.file2text(**kwargs))
        except Exception as e:
            msg = f"从文件 {file.kb_name}/{file.filename} 加载文档时出错：{e}"
            return False, (file.kb_name, file.filename, msg)

    kwargs_list = []
    for i, file in enumerate(files):
        kwargs = {}
        try:
            if isinstance(file, tuple) and len(file) >= 2:
                filename = file[0]
                kb_name = file[1]
                file = KnowledgeFile(
                    filename=filename, knowledge_base_name=kb_name)
            elif isinstance(file, dict):
                filename = file.pop("filename")
                kb_name = file.pop("kb_name")
                kwargs.update(file)
                file = KnowledgeFile(
                    filename=filename, knowledge_base_name=kb_name)
            # kwargs["file"] = file
            kwargs['text_splitter'] = splitter
            kwargs_list.append(kwargs)
        except Exception as e:
            yield False, (kb_name, filename, str(e))

    for result in run_in_thread_pool(func=file2docs, params=kwargs_list, pool=pool):
        yield result


check_vdb_init_status = {}


def sync_kbot_records(
        knowledge_base_name: str = Body(..., examples=["samples"]),
        stub: str = Body('stub', examples=[
            "for json body, no need to input "]),
) -> ORJSONResponse:
    json = queryFilesByKB(knowledge_base_name)
    return ORJSONResponse(json)


def check_vector_store_embedding_progress(
        knowledge_base_name: str = Body(..., examples=["samples"]),
        stub: str = Body('stub', examples=[
            "for json body, no need to input "]),
) -> ORJSONResponse:
    return ORJSONResponse(
        check_vdb_init_status[knowledge_base_name] if knowledge_base_name in check_vdb_init_status else "")


def delete_kb(knowledge_base_name: str = Body(..., examples=["samples"]),
              stub: str = Body('stub', examples=["for json body, no need to input "])):
    if knowledge_base_name is None or knowledge_base_name.strip() == "":
        return BaseResponse(code=404, msg="知识库名称不能为空，请重新填写知识库名称")
    files = list_files_from_folder(knowledge_base_name)
    # delete vectors and file record in sqlite
    delete_docs(knowledge_base_name, files)
    vs, kb = get_vs_from_kb(knowledge_base_name)
    # if 'oracle' == kb.vs_type:
    #    vs.delete_collection()
    if delete_kb_from_db(knowledge_base_name=knowledge_base_name):
        kbPath = get_kb_path(knowledge_base_name)
        delete_folder(kbPath)
        batchInfo = KnowledgeBatchInfo(kb_name=knowledge_base_name)
        delete_allbatchInfo_in_one_kb(batchInfo)

        return BaseResponse(code=200, msg="删除成功")
    else:
        return BaseResponse(code=404, msg="删除失败")


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
            kb, chunk_size, chunk_overlap = queryEmbedSettingByFile(src_file)
            # fileEmbeddingSetting = FileEmbeddingSetting(filepath=src_file, kb=kb, chunk_size=chunk_size,
            #                                             chunk_overlap=chunk_overlap)
            # add_file_embedding_setting_to_db(fileEmbeddingSetting)


def recreate_vector_store(
        knowledge_base_name: str = Body(..., examples=["samples"]),
        embed_model: str = Body(EMBEDDING_MODEL),
        vector_store_type: str = Body('faiss', description="vs type"),
):
    '''
    recreate vector store from the content.
    this is usefull when user can copy files to content folder directly instead of upload through network.
    '''
    kb = load_kb_from_db(knowledge_base_name)
    kbInfo = kb.kb_info
    contentPath = get_content_root(knowledge_base_name)
    dest_dir = tempfile.mkdtemp()

    copy_to_dest_dir(contentPath, dest_dir)

    delete_kb(knowledge_base_name)
    create_kb(knowledge_base_name=knowledge_base_name,
              embed_model=embed_model,
              vector_store_type=vector_store_type,
              knowledge_base_info=kbInfo
              )
    copy_to_dest_dir(dest_dir, contentPath)

    def output():
        check_vdb_init_status[knowledge_base_name] = 'processing'
        msg_array = []

        kbPath = get_kb_path(knowledge_base_name)
        if not os.path.exists(kbPath):
            yield {"code": 404, "msg": f"未找到知识库 ‘{knowledge_base_name}’"}
        else:
            # for faiss
            # if os.path.exists(vsPath):
            #     shutil.rmtree(vsPath)

            # os.makedirs(vsPath)
            # vector_store = init_vs(knowledge_base_name, embed_model, vector_store_type)
            # delete_files_from_db(knowledge_base_name)
            files = list_files_from_folder(knowledge_base_name)

            kb_files = [KnowledgeFile(
                filepath=file, knowledge_base_name=knowledge_base_name) for file in files]
            i = 0
            for kb_file in kb_files:
                kb, chunk_size, chunk_overlap = queryEmbedSettingByFile(
                    file_path=kb_file.filepath)

                msg = ''
                progress_obj = {
                    "total": len(files),
                    "finished": i,
                    "current_document": kb_file.filepath,
                    'details': msg_array,
                    'chunk_size': chunk_size,
                    'chunk_overlap': chunk_overlap
                }
                try:
                    text_splitter = makeSplitter(chunk_size, chunk_overlap)
                    chunkDocuments = kb_file.rebuild_file2text(text_splitter)
                    store_vectors_by_batch(
                        knowledge_base_name, kb.embed_model, chunkDocuments)
                    status = add_file_to_db(kb_file)

                    if not status:
                        msg = f"添加文件‘{kb_file.filepath}’到sqlite‘{knowledge_base_name}’时出错 。已跳过。"

                except:
                    msg = f"添加文件‘{kb_file.filepath}’ as vectors ‘{knowledge_base_name}’时出错 。已跳过。"

                i = i + 1
                progress_obj['details'].append(
                    {'file_name': kb_file.filename, 'msg': msg})
                progress_obj['finished'] = i
                check_vdb_init_status[knowledge_base_name] = progress_obj
                yield json.dumps(check_vdb_init_status[knowledge_base_name], ensure_ascii=False)

    return StreamingResponse(output(), media_type="text/event-stream")


def query_in_kb(
        user: str = Body(..., description="current user", examples=['Demo']),
        ask: str = Body(..., description="query", examples=["What's kyc"]),
        kb_name: str = Body(..., description="knowledge_base_name", examples=[
            "samples"]),
        rerankerModel: str = Body(
            'bgeReranker', description='which reranker model'),
        reranker_topk: int = Body(2, description='reranker_topk'),
        score_threshold: float = Body(
            0.6, description='reranker score threshold'),
        vector_store_limit: int = Body(
            10, description='the limit of query from vector db'),
        search_type=Body(
            'vector', description='the type of search. eg. vector, full_text, hybrid'),

):
    # 1.初始化配置参数
    settings = SimpleNamespace()

    settings.rerankerModel = rerankerModel
    settings.reranker_topk = reranker_topk
    settings.score_threshold = score_threshold
    settings.vector_store_limit = vector_store_limit
    settings.search_type = search_type
    user_settings[user] = settings

    logger.info(f"##1).完成配置参数初始化:{get_cur_time()}##")

    # 2.获取向量检索结果以及rerank结果
    status: str = "success"
    err_msg: str = ""
    vector_res_arr = []
    vector_res_arr, _ = makeSimilarDocs(ask, kb_name, user)
    logger.info("##2)完成获取向量检索结果以及rerank结果。##")

    # 将结果对象列表转换为JSON数组
    result_str = json.dumps(
        [{"content": p.content, "source": p.source,
          "score": float(p.score)} for p in vector_res_arr],
        ensure_ascii=False)
    result_list = json.loads(format_llm_response(result_str))
    logger.info(f"##3).完成LLM结果处理:{get_cur_time()}##")
    return VectorSearchResponse(
        data=result_list,
        status=status,
        err_msg=err_msg
    )


def fulltext_search(question, kb_name, user):
    _, kb = get_vs_from_kb(kb_name)
    settings = user_settings.get(user)
    fullTextDocsWithScore = []
    if kb.vs_type == 'opensearch':
        pass  # todo
    elif kb.vs_type == 'faiss':
        pass
    elif kb.vs_type == 'oracle' or kb.vs_type == 'adb':
        fullTextDocsWithScore = oracle_fulltext_helper.search_with_score_by_text(question, kb_name,
                                                                                 settings.vector_store_limit,
                                                                                 kb.vs_type)

    return fullTextDocsWithScore


def makeSimilarDocs(question, kb_name, user):
    llm_context = ""
    vector_res_arr: List[AskResponseData] = []
    # kb = load_kb_from_db(kb_name)
    settings = user_settings.get(user)

    similar_doc_with_socres = get_docs_with_scores(question, kb_name, user)
    contentRoot = get_content_root(kb_name)
    for obj in similar_doc_with_socres:
        doc = obj[0]
        doc_content = doc.page_content
        doc_source = doc.metadata.get("source")
        doc_score = obj[1]
        logger.info(
            f"##doc_score: {doc_score}, score_threshold: {settings.score_threshold}, doc_source: {doc_source}")
        # doc_score（向量相识度/Reranker）统一成越大，越相似，所以需要大于配置的阈值才显示。

        if doc_score >= settings.score_threshold:
            if doc_source.startswith('http') or doc_source.startswith('https'):
                llm_context = llm_context + "Url: " + \
                              doc_source + ' :\n' + doc_content + '\n\n'
            else:
                file_original_path = Path(doc_source)
                llm_context = llm_context + "Document " + \
                              file_original_path.stem + ' :\n' + doc_content + '\n'

                # 获取Vector DB中的数据源
                # Remove the prefix path
                filename = file_original_path.relative_to(Path(contentRoot))
                if config.DOC_VIEWER_FLAG == 'Y':
                    doc_source = config.http_doc_viewer + f'{kb_name}/content/{filename}'
                else:
                    if not config.http_prefix.endswith('/'):
                        config.http_prefix += '/'
                    doc_source = config.http_prefix + f'knowledge_base/download_doc?knowledge_base_name={kb_name}&file_name={filename}'
            askRes = AskResponseData(doc_content, doc_source, doc_score)
            vector_res_arr.append(askRes)
    return vector_res_arr, llm_context


def load_bge_reranker_large_model(model_path: str = config.BGE_RERANK_PATH) -> FlagReranker:
    reranker = FlagReranker(model_path,
                            use_fp16=True)  # Setting use_fp16 to True speeds up computation with a slight performance degradation
    return reranker


def reRankVectorResult(query: str, vectorResut: List[Document],  user: str) -> List[Document]:
    settings = user_settings.get(user)
    top_k=settings.reranker_topk
    if settings.rerankerModel == 'bgeReranker':
        bge_reranker_large_model = load_bge_reranker_large_model()
        reRankResult = reRankVectorResultByBgeReranker(bge_reranker_large_model, query, vectorResut,
                                                       top_k)
    elif settings.rerankerModel == 'cohereReranker':
        reRankResult = reRankVectorResultByCohere(query, vectorResut, top_k)
    elif settings.rerankerModel == 'disableReranker':
        reRankResult = vectorResut
    return reRankResult


# 对向量数据库结果使用Cohere Rerank模型进行Rerank
def reRankVectorResultByCohere(query: str, vectorResut: List[Document], top_k: int = 3) -> List[Document]:
    # 1.初始化Cohere客户端
    api_key = llm_keys.cohere_api_key
    co = cohere.Client(api_key)
    # 2.获取Rerank所需要的数据格式
    src_docs = []
    for obj in vectorResut:
        doc = obj[0]
        doc_content = doc.page_content
        src_docs.append(doc_content)
    # 3.执行Rerank
    reRankDocs = co.rerank(query=query, documents=src_docs, top_n=top_k,
                           model='rerank-multilingual-v3.0',
                           return_documents=True)  # Change top_n to change the number of results returned. If top_n is not passed, all results will be returned.
    # 4.获取Rerank结果
    reRankResult = []
    for _, r in enumerate(reRankDocs.results):
        reRankResult.append(
            (Document(page_content=r.document.text, metadata=vectorResut[r.index][0].metadata), r.relevance_score))
    return reRankResult


def reRankVectorResultByBgeReranker(rerankModel: FlagReranker, query: str, vectorResut: List[Document],
                                    top_k: int = 3) -> List[Document]:
    # 1.获取Rerank所需要的数据格式
    src_docs = []
    for obj in vectorResut:
        # logger.info("index:",obj)
        doc = obj[0]
        doc_content = doc.page_content
        src_docs.append([query, doc_content])

    # 2.执行Rerank
    # scores = rerankModel.compute_score([['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']])
    scores = rerankModel.compute_score(src_docs,normalize=True)
    # 3.获取Rerank结果
    reRankResult = []
    index: int = 0
    for obj in vectorResut:
        doc = obj[0]
        reRankResult.append((doc, scores[index]))
        index = index + 1
        # 只返回top_k的结果
        if index >= top_k:
            break
    # 4.按照rerank的score排序，从大到小排序
    reRankResult = sorted(
        reRankResult, key=lambda tupleobj: tupleobj[1], reverse=True)
    return reRankResult


def vectorOpenSearchWrapper(query, knowledge_base_name, user):
    settings = user_settings.get(user)
    vector_store, _ = get_vs_from_kb(knowledge_base_name)

    return vector_store.similarity_search_with_score_by_vector(query, k=settings.vector_store_limit)


def vectorSearchWrapper(query, knowledge_base_name, user):
    settings = user_settings.get(user)
    vector_store, _ = get_vs_from_kb(knowledge_base_name)

    return vector_store.similarity_search_with_relevance_scores(query, k=settings.vector_store_limit)


def get_docs_with_scores(query, knowledge_base_name, user) -> List[Document]:
    vector_store, kb = get_vs_from_kb(knowledge_base_name)
    settings = user_settings.get(user)
    search_functions = {
        'opensearch': {
            'vector': vectorOpenSearchWrapper,
            # 'fulltext': opensearch_fulltext_search
        },
        'oracle': {
            'vector': vectorSearchWrapper,
            'fulltext': fulltext_search
        },
        'adb': {
            'vector': vectorSearchWrapper,
            'fulltext': fulltext_search
        },
        'faiss': {
            'vector': vectorSearchWrapper,
        },
        'heatwave': {
            'vector': vectorSearchWrapper,
        },
    }
    # if kb.vs_type == 'opensearch':
    #     embedding = config.EMBEDDING_DICT[kb.embed_model]
    #     query_vector = embedding.embed_query(query)
    #     # similarity_search_with_score_by_vector,这个结果是越小，越相似
    #     similar_doc_with_scores = search_functions.get(kb.vs_type+"_"+settings.search_type)(query_vector,
    #                                                                                 k=settings.vector_store_limit)
    # else:
    #     # max_marginal_relevance_search_with_score_by_vector这个是越小，越相似
    #     # similar_doc_with_scores = vector_store.max_marginal_relevance_search_with_score_by_vector(
    #     #    query_vector, k=config.vector_store_limit)
    #     # similarity_search_with_relevance_scores是越大，越相似，取值是[0,1]
    #     similar_doc_with_scores = search_functions.get(kb.vs_type+"_"+settings.search_type) (
    #        query, k=settings.vector_store_limit)

    if settings.search_type == 'hybrid':  # 如果是hyrid检索，则是向量检索+全文检索的组合，如果全文检索的结果和向量结果一样，则优先获取向量结果
        similar_doc_with_scores = search_functions[kb.vs_type]['vector'](
            query, knowledge_base_name, user)
        fullTextDocsWithScore = search_functions[kb.vs_type]['fulltext'](
            query, knowledge_base_name, user)
        similar_doc_with_scores = merge_search_results(
            similar_doc_with_scores, fullTextDocsWithScore)
    else:
        similar_doc_with_scores = search_functions[kb.vs_type][settings.search_type](
            query, knowledge_base_name, user)

    logger.info(
        f"####检索结果kb.vs_type:{kb.vs_type},search_type:{settings.search_type},len(similar_doc_with_scores):{len(similar_doc_with_scores)}, vector_store_limit:{settings.vector_store_limit} reranker_topk:{settings.reranker_topk}")
    # for obj in similar_doc_with_scores:
    #   doc = obj[0]
    #   doc_source = doc.metadata.get("source")
    #   doc_score = obj[1]
    #   logger.info(f"##向量检索结果明细,doc_score: {doc_score}, doc_source: {doc_source}")
    if len(similar_doc_with_scores) > 1:
        # Rereank的结果是越大，越相似
        similar_doc_with_scores = reRankVectorResult(
            query, similar_doc_with_scores,  user)
    return similar_doc_with_scores


def get_kb_info(knowledge_base_name: str = Query(..., description="知识库名称", examples=["bank"])):
    kb = load_kb_from_db(knowledge_base_name)
    content_root = get_content_root(knowledge_base_name)
    dd = {'content_root': content_root,
          'info': kb.kb_info,
          'files_count': kb.file_count,
          'embedding_model': kb.embed_model,
          'vs_type': kb.vs_type,
          'create_time': kb.create_time
          }
    if kb is not None:
        return BaseResponse(code=200, data=dd)
    else:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}", data="no info")


def download_doc(
        knowledge_base_name: str = Query(..., description="知识库名称", examples=[
            "bank"]),
        file_name: str = Query(..., description="文件名称", examples=["test.txt"]),
):
    '''
    下载知识库文档
    '''

    if knowledge_base_name is None or knowledge_base_name.strip() == "":
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    try:
        kb_file = KnowledgeFile(filename=file_name,
                                knowledge_base_name=knowledge_base_name)
        content_disposition_type = None
        if os.path.exists(kb_file.filepath):
            return FileResponse(
                path=kb_file.filepath,
                filename=kb_file.filename,
                media_type="multipart/form-data",
                content_disposition_type=content_disposition_type,
            )
    except Exception as e:
        msg = f"{kb_file.filename} 读取文件失败，错误信息是：{e}"
        return BaseResponse(code=500, msg=msg)

    return BaseResponse(code=500, msg=f"{kb_file.filename} 读取文件失败")


def create_kb(knowledge_base_name: str = Body(..., examples=["samples"]),
              knowledge_base_info: str = Body(..., examples=[
                  "this is about bank"]),
              vector_store_type: str = Body("faiss"),
              embed_model: str = Body(EMBEDDING_MODEL),
              ) -> BaseResponse:
    # Create selected knowledge base

    if knowledge_base_name is None or knowledge_base_name.strip() == "":
        return BaseResponse(code=404, msg="knowledge_base_name should not be empty")

    kb = checkIfKBExists(knowledge_base_name)
    if kb is not None:
        return BaseResponse(code=500, msg=f"had the same name {knowledge_base_name}")
    try:
        create_dir(knowledge_base_name)
        add_kb_to_db(knowledge_base_name, knowledge_base_info,
                     vector_store_type, embed_model)
        init_vs(knowledge_base_name, embed_model, vector_store_type)
    except Exception as e:
        msg = f"创建知识库出错： {e}"
        logger.error(msg)

        return BaseResponse(code=500, msg=msg)

    return BaseResponse(code=200, msg=f"successfully added knowledge_base {knowledge_base_name}")
