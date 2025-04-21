import argparse
from pathlib import Path
from tqdm import tqdm
from kb_api import checkIfKBExists,  KnowledgeFile, store_vectors_by_batch, KnowledgeBatchInfo, \
    add_file_to_db, add_batchInfo_to_db
from util import makeSplitter, get_content_root
### init logging
from loguru import logger
from typing import List, Union, Dict
from langchain_core.documents import Document
logger.add("offlineUpload.log", rotation="5 MB")


import uuid
import config

# Define the prompt template
prompt_text = """
You are an assistant tasked with summarizing tables and text particularly for semantic retrieval.
These summaries will be embedded and used to retrieve the raw text or table elements.
Give a detailed summary of the table or text below that is well optimized for retrieval.
For any tables also add in a one-line description of what the table is about besides the summary.
Do not add additional words like "Summary:" etc.

Table or text chunk:
{element}
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
import copy

prompt = ChatPromptTemplate.from_template(prompt_text)
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

summary_model = config.MODEL_DICT.get(config.SUMMARY_MODEL)

# Define the summary chain
summarize_chain = (
        {"element": RunnablePassthrough()}
        | prompt
        | summary_model
        | StrOutputParser()  # Extracts the response as text
)

def add_summaries_to_langchain_documents(data: List[Document]):
    docs = []
    tables = []
    images = []
    for doc in data:
        file_type = doc.metadata.get('filetype', 'non_pdf')
        if file_type == 'application/pdf':
            doc.metadata['page'] = doc.metadata.get('page_number', 0)
            doc.metadata["chunk_id"] = str(uuid.uuid4())
            if doc.metadata.get('category') == 'Table':
                doc.metadata["chunk_category"] = "SOURCE_TABLE"
                tables.append(doc)
            elif doc.metadata.get('category') == 'CompositeElement':
                doc.metadata["chunk_category"] = "SOURCE_DOCS"
                docs.append(doc)
            elif doc.metadata.get('category') == 'Image':
                doc.metadata["chunk_category"] = "SOURCE_IMAGE"
                images.append(doc)
        else:
            docs.append(doc)

    summary_chunk_list = []
    src_chunk_list = docs + tables + images

    for doc in src_chunk_list:
        file_type = doc.metadata.get('filetype', 'non_pdf')
        if file_type == 'application/pdf':
            summary = summarize_chain.invoke(doc.page_content)
            # Update previous_chunk for the next iteration
            newdoc = copy.deepcopy(doc)
            newdoc.page_content = summary
            newdoc.metadata["chunk_id"] = str(uuid.uuid4())
            newdoc.metadata["src_chunk_id"] = doc.metadata["chunk_id"]
            newdoc.metadata["chunk_category"] = doc.metadata["chunk_category"].replace('SOURCE', 'SUMMARY')
            summary_chunk_list.append(newdoc)
    return summary_chunk_list + src_chunk_list

def offlineUploadRuntime(knowledge_base_name, batch_name,chunk_size, chunk_overlap,ocr_lang,multivector):
    if knowledge_base_name is None or knowledge_base_name.strip() == "":
        raise ValueError("knowledge_base_name should not be empty!")
    kb = checkIfKBExists(knowledge_base_name)
    if kb is None:
        raise ValueError(f"knowledge_base  {knowledge_base_name} not found")

    failed_files = {}
    # file_names = list()
    # 先将上传的文件保存到磁盘

    contentRoot = get_content_root(knowledge_base_name)
    filesDir = Path(contentRoot) /batch_name
    all_real_files = [file for file in filesDir.rglob('*') if file.is_file() ]
    saved_disk_files = [str(file.relative_to(filesDir)) for file in all_real_files]

    kb_files = [KnowledgeFile(filename=str(Path(batch_name) / file), knowledge_base_name=knowledge_base_name) for file
                in saved_disk_files]
    logger.info('vectorizing uploaded docs...')
    for kb_file in tqdm(kb_files):
        try:
            text_splitter = makeSplitter(chunk_size, chunk_overlap)
            chunkDocuments = kb_file.file2text(text_splitter, ocr_lang)
            if 'summary' in multivector:
                chunkDocuments = add_summaries_to_langchain_documents(chunkDocuments)
            store_vectors_by_batch(knowledge_base_name, kb.embed_model, chunkDocuments)
            kb_file.batch_name = batch_name

            kb_file.chunk_overlap = chunk_overlap
            kb_file.chunk_size = chunk_size
            kb_file.knowledge_base_name = knowledge_base_name
            batchInfo = KnowledgeBatchInfo(batch_name=batch_name,
                                           kb_name=knowledge_base_name,
                                           chunk_size=chunk_size,
                                           chunk_overlap=chunk_overlap)

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
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='offline upload cmd',
                                     description='About hub knowledge based apis exposed as  rest-svc,  ')
    parser.add_argument("--knowledge_base_name", type=str,  )
    parser.add_argument("--batch_name", type=str,  )
    parser.add_argument("--chunk_size", type=int, default=250)
    parser.add_argument("--chunk_overlap", type=int, default=22)
    parser.add_argument("--ocr_lang", type=str,default='en')
    parser.add_argument("--multivector", type=str,default='')
    # 初始化消息
    args = parser.parse_args()

    offlineUploadRuntime( args.knowledge_base_name,
                          args.batch_name,
                          args.chunk_size,
                          args.chunk_overlap,
                          args.ocr_lang,
                          args.multivector
            )
