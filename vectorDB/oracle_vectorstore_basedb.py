""" 
Description: 
 - Implement Oracle VectorStore for Oracle Base Database.

History:
 - 2024/09/07 by Hysun (hysun.he@oracle.com): Created
"""

import oracledb
from config import config
from vectorDB.oracle_vectorstore_abstract import AbstractOracleVS
from loguru import logger

class OracleBaseDbVS(AbstractOracleVS):
    """Oracle VectorStore for Oracle Base Database.
    Example:
        vector_store: VectorStore = OracleBaseDbVS(
            collection_name="Hysun_Test_OracleBaseDb",
            embedding_function=embedding_model,
            pre_delete_collection=True,
        )
    """

    def __init__(self, *args, **kwargs):
        super(OracleBaseDbVS, self).__init__(*args, **kwargs)

    def connect(self) -> oracledb.Connection:
        logger.info("##connecting to dbcs 23ai ###")
        return oracledb.connect(dsn=config.ORACLE_AI_VECTOR_CONNECTION_STRING)
