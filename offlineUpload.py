import argparse
from pathlib import Path

from tqdm import tqdm

from kb_api import checkIfKBExists,  KnowledgeFile, store_vectors_by_batch, KnowledgeBatchInfo, \
    add_file_to_db, add_batchInfo_to_db
from util import makeSplitter, get_content_root

### init logging
from loguru import logger
logger.add("offlineUpload.log", rotation="5 MB")



def offlineUploadRuntime(knowledge_base_name, batch_name,chunk_size, chunk_overlap,ocr_lang):
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
    logger.info('vectorize uploaded docs...')
    for kb_file in tqdm(kb_files):
        try:
            text_splitter = makeSplitter(chunk_size, chunk_overlap)
            chunkDocuments = kb_file.file2text(text_splitter, ocr_lang)
            store_vectors_by_batch(knowledge_base_name, kb.embed_model, chunkDocuments)
            kb_file.batch_name = batch_name

            kb_file.chunk_overlap = chunk_overlap
            kb_file.chunk_size = chunk_size
            kb_file.knowledge_base_name = knowledge_base_name
            batchInfo = KnowledgeBatchInfo(batch_name=batch_name,
                                           kb_name=knowledge_base_name,
                                           chunk_size=chunk_size,
                                           chunk_overlap=chunk_overlap)

            # fileEmbeddingSetting = FileEmbeddingSetting(batch_name =batch_name,
            #                                             file_path=kb_file.filepath,
            #                                             chunk_overlap=chunk_overlap,
            #                                             chunk_size=chunk_size )
            # add_file_embedding_setting_to_db(fileEmbeddingSetting)
            # status = add_file_to_db(kb_file)

            # if not status:
            # msg = f"adding file ‘{kb_file.filename}’ to sqlite db failed：skip。"
            # failed_files[kb_file.filename] = msg
            # logger.info(msg)
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
    # 初始化消息
    args = parser.parse_args()

    offlineUploadRuntime( args.knowledge_base_name,
                          args.batch_name,
                          args.chunk_size,
                          args.chunk_overlap,
                          args.ocr_lang
            )
