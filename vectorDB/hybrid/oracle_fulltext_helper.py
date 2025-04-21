""" 
Description: 
 - Query helper for hybird search (Oracle Text Search + Vector Search).

pip requirement:
    pip install jieba==0.42.1

#先用dba用户赋予vector_prd用户可执行权限
grant execute on ctxsys.ctx_ddl to vector_prd;

#在普通用户执行如下步骤，创建索引。（中文）
exec ctx_ddl.create_preference('chinese_lexer','chinese_vgram_lexer');
CREATE INDEX  "KM_TEXTSEARCH_IDX" ON  kbot_oracle_embeddings("DOCUMENT") INDEXTYPE IS "CTXSYS"."CONTEXT" parameters ('lexer chinese_lexer');

#在普通用户执行如下步骤，创建索引。（英文）
exec ctx_ddl.create_preference('english_lexer','basic_lexer');
CREATE INDEX  "KM_TEXTSEARCH_IDX" ON  kbot_oracle_embeddings("DOCUMENT") INDEXTYPE IS "CTXSYS"."CONTEXT" parameters ('lexer english_lexer');

History:
 - 2024/08/02 by Hysun (hysun.he@oracle.com): Created.
   2024/08/15 by Hysun: Bug fix: handle punctuation in the query.  
"""

import oracledb
import config
from oracledb import Connection
from loguru import logger
from langchain_core.documents import Document

from typing import Any, List, Tuple, Optional

import subprocess
import sys
import re

try:
    import jieba
except ImportError:
    logger.info("### Install Chinese tokenizer.")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "jieba"])
finally:
    import jieba


def query_by_text_search(
    query: str, collection_name: str, topk: int = 10, dbtype: str = "oracle"
) -> List[Any]:
    """Perform Oracle Text search

    Args:
        query (str): _description_
        collection_name (str): _description_
        topk (int, optional): _description_. Defaults to 10.

    Returns:
        List[Any]: _description_
    """
    logger.info(f"### dbtype: {dbtype} Oracle-text search: {collection_name} - {topk}: {query}")
    query = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", query)).strip()
    tokens = jieba.cut(query)
    tokens_string = " ".join(tokens)
    logger.info(f"### Tokenized query: {tokens_string}")

    sql = f"""
    SELECT 
        id,
        document,
        metadata,
        score(9) as score
    FROM kbot_oracle_embeddings
    WHERE collection_name = :1
        and (chunk_category is null OR chunk_category like 'SOURCE_%')
        and contains(document, regexp_replace(:2,'\\W+', ' ACCUM '), 9) > 0
    ORDER BY score DESC
    FETCH FIRST {topk} ROWS ONLY
    """

    json_results = []
    if dbtype == "oracle":
        connection: Connection = oracledb.connect(
            dsn=config.ORACLE_AI_VECTOR_CONNECTION_STRING
        )
    elif dbtype == "adb":
        connection: Connection = oracledb.connect(
            user=config.ADW_VECTOR_SEARCH_USER,
            password=config.ADW_VECTOR_SEARCH_PASSWORD,
            dsn=config.ADW_VECTOR_SEARCH_DSN,
            wallet_location=config.ADW_VECTOR_SEARCH_WALLET_LOCATION,
            config_dir=config.ADW_VECTOR_SEARCH_WALLET_LOCATION,
            wallet_password=config.ADW_VECTOR_SEARCH_WALLET_PASSWORD,
        )
    else:
        raise NotImplementedError(f"! DB type is not supported: {dbtype}")

    with connection:
        with connection.cursor() as cursor:
            for result in cursor.execute(sql, [collection_name, tokens_string]):
                json_results.append(
                    {
                        "collection_id": collection_name,
                        "document": result[1].read(),
                        "cmetadata": result[2],
                        "custom_id": "",
                        "uuid": result[0],
                        "score": result[3],
                    }
                )

    logger.info(f"### Oracle text search got records: {len(json_results)}")
    return json_results


def _results_to_docs_and_scores(results: Any) -> List[Tuple[Document, float]]:
    """Return docs and scores from results.

    Returns:
        _type_: _description_
    """
    docs = [
        (
            Document(
                page_content=result["document"],
                metadata=result["cmetadata"],
            ),
            result["score"],
        )
        for result in results
    ]
    return docs


def search_with_score_by_text(
    query: str, collection_name: str, topk: int = 10, dbtype: str = "oracle"
):
    """Search by Oracle text.

    Args:
        query (str): _description_
        collection_name (str): _description_
        topk (int, optional): _description_. Defaults to 10.

    Returns:
        _type_: _description_
    """
    try:
        results = query_by_text_search(query, collection_name, topk, dbtype)
        return _results_to_docs_and_scores(results)
    except Exception as e:
        logger.error(f"### Oracle text search failed {str(e)}")
        return []


def merge_search_results(
    docs_by_vector: Optional[List[Tuple[Document, float]]],
    docs_by_text: Optional[List[Tuple[Document, float]]],
):
    """Union the two query result lists with eliminating duplicate records.

    Args:
        docs_by_vector (List[Tuple[Document, float]]): _description_
        docs_by_text (List[Tuple[Document, float]]): _description_

    Returns:
        _type_: _description_
    """
    docs_by_vector = [] if not docs_by_vector else docs_by_vector
    docs_by_text = [] if not docs_by_text else docs_by_text
    logger.info(f"### Merge result: {len(docs_by_vector)} | {len(docs_by_text)} ")

    merged_results: List[Tuple[Document, float]] = [] + docs_by_vector
    vdocs = [doc.page_content for doc, _ in docs_by_vector]
    for doc, score in docs_by_text:
        content = doc.page_content
        if content not in vdocs:
            merged_results.append((doc, score))

    logger.info(f"### Merged result size: {len(merged_results)}")
    return merged_results
