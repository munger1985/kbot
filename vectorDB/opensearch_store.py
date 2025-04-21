
import config
from loguru import logger
from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy import OpenSearch
from langchain_core.documents import Document

# 创建连接

opensearch_pyclient= OpenSearch(
            opensearch_url = config.OCI_OPEN_SEARCH_URL,
            http_auth =(config.OCI_OPEN_SEARCH_USER,config. OCI_OPEN_SEARCH_PASSWD),
            verify_certs=False,
            use_ssl=True
        )


def delete_doc(kb_file):
    knowledge_base_name=kb_file.knowledge_base_name
    # 定义查询条件
    query = {
        "query": {
            "term": {
                "metadata.source.keyword": kb_file.filepath
            }
        },
        "_source": False,  # 不返回文档内容，只返回元数据
        "fields": ["_id"]   # 只返回文档ID
    }

    response = opensearch_pyclient.search(body=query, index=knowledge_base_name)

    # 提取文档ID
    doc_ids = [hit['_id'] for hit in response['hits']['hits']]
    logger.info("Document IDs to be deleted:", doc_ids)
    for doc_id in doc_ids:
        response = opensearch_pyclient.delete(index=knowledge_base_name, id=doc_id)



def opensearch_fulltext_wrapper(question, kb_name):
        # 查询文档
    query = {
    "query": {
        "match": {
        "text": question
        }
    }
    }

    results = opensearch_pyclient.search(
        index = kb_name,
        body = query
    )
    docs = [
        (
            Document(
                page_content=result["_source"]['text'],
                metadata=result["_source"]['metadata'],
            ),
            result["_score"],
        )
        for result in results['hits']['hits']
    ]
    logger.info(docs)
    return docs
