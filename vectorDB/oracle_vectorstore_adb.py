""" 
Description: 
 - Implement Oracle VectorStore for ADB.

History:
 - 2024/09/07 by Hysun (hysun.he@oracle.com): Created
"""

import oracledb
import config
from vectorDB.oracle_vectorstore_abstract import AbstractOracleVS
from loguru import logger

class OracleAdbVS(AbstractOracleVS):
    """Oracle VectorStore for ADB
    Example:
        vector_store: VectorStore = OracleAdbVS(
            collection_name="Hysun_Test",
            embedding_function=embedding_model,
            pre_delete_collection=True,
        )
    """

    def __init__(self, *args, **kwargs):
        super(OracleAdbVS, self).__init__(*args, **kwargs)

    def connect(self) -> oracledb.Connection:
        #logger.info("##connecting to ADW 23ai ###")
        return oracledb.connect(
            user=config.ADW_VECTOR_SEARCH_USER,
            password=config.ADW_VECTOR_SEARCH_PASSWORD,
            dsn=config.ADW_VECTOR_SEARCH_DSN,
            wallet_location=config.ADW_VECTOR_SEARCH_WALLET_LOCATION,
            config_dir=config.ADW_VECTOR_SEARCH_WALLET_LOCATION,
            wallet_password=config.ADW_VECTOR_SEARCH_WALLET_PASSWORD,
        )
