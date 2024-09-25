from __future__ import annotations
from loguru import logger
import os
import config
import asyncio
import contextlib
import enum
import json
import logging
import uuid
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
)
from datetime import datetime
import numpy as np
import oracledb
import sqlalchemy
from sqlalchemy import delete
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Session
from langchain_community.vectorstores.utils import maximal_marginal_relevance
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import (
    DistanceStrategy,
)
try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base

DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.COSINE

Base = declarative_base()  # type: Any

_LANGCHAIN_DEFAULT_COLLECTION_NAME = "langchain"

def _get_distance_function(distance_strategy: DistanceStrategy) -> str:
    # Dictionary to map distance strategies to their corresponding function
    # names
    distance_strategy2function = {
        DistanceStrategy.EUCLIDEAN_DISTANCE: "EUCLIDEAN",
        DistanceStrategy.DOT_PRODUCT: "DOT",
        DistanceStrategy.COSINE: "COSINE",
    }

    # Attempt to return the corresponding distance function
    if distance_strategy in distance_strategy2function:
        return distance_strategy2function[distance_strategy]

    # If it's an unsupported distance strategy, raise an error
    raise ValueError(f"Unsupported distance strategy: {distance_strategy}")

class BaseModel(Base):
    """Base model for the SQL stores."""

    __abstract__ = True
    uuid = sqlalchemy.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


def _results_to_docs(docs_and_scores: Any) -> List[Document]:
    """Return docs from docs and scores."""
    return [doc for doc, _ in docs_and_scores]


OracleConnectionPool = 1


def get_connection_from_pool(func):
    def wrapper(self, *args, **kwargs):
        # Get a connection from the pool
        logger.info(22, func)

        global OracleConnectionPool
        # while True:
        try:
            logger.info('11 break')

            self._conn = OracleConnectionPool.acquire()
            logger.info('break')
            # break

        except Exception as e:

            logger.info(e)
            import time
            time.sleep(1)
        try:
            # Call the function
            logger.info(2255)

            return func(self, *args, **kwargs)


        except Exception as e:
            logger.info(e)
        finally:
            # self._conn.close()
            logger.info(333444)

            # Release the connection back to the pool
            OracleConnectionPool.release(self._conn)
            logger.info(444)

    return wrapper


class OracleAIVector(VectorStore):
    """`Postgres`/`OracleAIVector` vector store.

    To use, you should have the ``pgvector`` python package installed.

    Args:
        connection_string: Postgres connection string.
        embedding_function: Any embedding function implementing
            `langchain.embeddings.base.Embeddings` interface.
        collection_name: The name of the collection to use. (default: langchain)
            NOTE: This is not the name of the table, but the name of the collection.
            The tables will be created when initializing the store (if not exists)
            So, make sure the user has the right permissions to create tables.
        distance_strategy: The distance strategy to use. (default: COSINE)
        pre_delete_collection: If True, will delete the collection if it exists.
            (default: False). Useful for testing.
        engine_args: SQLAlchemy's create engine arguments.

    Example:
        .. code-block:: python

            from mylangchain.vectorstores import OracleAIVector
            from langchain.embeddings.openai import OpenAIEmbeddings

            CONNECTION_STRING = "vectordemo/welcome1@146.235.233.91:1521/pdb1.sub08030309530.justinvnc1.oraclevcn.com"
            COLLECTION_NAME = "state_of_the_union_test"
            embeddings = OpenAIEmbeddings()
            vectorestore = OracleAIVector.from_documents(
                embedding=embeddings,
                documents=docs,
                collection_name=COLLECTION_NAME,
                connection_string=CONNECTION_STRING,
            )


    """

    def __init__(
            self,
            connection_string: str,
            embedding_function: Embeddings,
            collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
            collection_metadata: Optional[dict] = None,
            distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
            pre_delete_collection: bool = False,
            logger: Optional[logging.Logger] = None,
            relevance_score_fn: Optional[Callable[[float], float]] = None,
            *,
            engine_args: Optional[dict[str, Any]] = None,
    ) -> None:
        self.connection_string = connection_string
        self.embedding_function = embedding_function
        self.collection_name = collection_name
        self.collection_metadata = collection_metadata
        self._distance_strategy = distance_strategy
        self.pre_delete_collection = pre_delete_collection
        self.logger = logger or logging.getLogger(__name__)
        self.override_relevance_score_fn = relevance_score_fn
        self.engine_args = engine_args or {}
        self.__post_init__()

    def __post_init__(
            self,
    ) -> None:
        """
        Initialize the store.
        """
        # self.make_pool_if_not_exists()
        self.connect()
        self.create_tables_if_not_exists()
        self.create_collection()

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_function

    def connect(self) -> oracledb.Connection:
        try:   
            #oracledb.init_oracle_client()
            self._conn = oracledb.connect(dsn=self.connection_string)
            # self._conn=self.pool.acquire()
            #print("###Get conn from DBCS 23ai conn####")
            return self._conn
        except Exception as e:
            logger.info(f"##conn exception:{e}")

    # @get_connection_from_pool
    def create_tables_if_not_exists(self) -> None:
        cursor = self._conn.cursor()

        cursor.execute(
            """
                CREATE TABLE IF NOT EXISTS langchain_oracle_collection (
                    name      VARCHAR2(200) NOT NULL,
                    cmetadata json NOT NULL,
                    uuid      VARCHAR2(200) NOT NULL,
                    CONSTRAINT loc_key_uuid PRIMARY KEY ( uuid )
                )
            """)

        cursor.execute(
            """
                CREATE TABLE IF NOT EXISTS langchain_oracle_embedding (
                    collection_id VARCHAR2(200) NOT NULL,
                    embedding     vector NOT NULL,
                    document      CLOB NOT NULL,
                    cmetadata     json NOT NULL,
                    custom_id     VARCHAR2(200) NOT NULL,
                    uuid          VARCHAR2(200) NOT NULL,
                    CONSTRAINT loe_key_uuid PRIMARY KEY ( uuid )
                )
            """)

        cursor.close()
        self._conn.commit()

    def drop_tables(self) -> None:
        cursor = self._conn.cursor()

        cursor.execute(
            """
                begin
                    execute immediate 'drop table langchain_oracle_embedding cascade constraints PURGE';
                    exception when others then if sqlcode <> -942 then raise; end if;
                end;
            """)

        cursor.execute(
            """
                begin
                    execute immediate 'drop table langchain_oracle_collection cascade constraints PURGE';
                    exception when others then if sqlcode <> -942 then raise; end if;
                end;
            """)

        self._conn.commit()

    def create_collection(self) -> None:
        if self.pre_delete_collection:
            self.delete_collection()
        self.get_or_create(
            name=self.collection_name, cmetadata=self.collection_metadata
        )

    def delete_collection(self) -> None:
        self.logger.debug("Trying to delete collection")
        collection = self.get_collection()
        if not collection:
            self.logger.warning("Collection not found")
            return

        cursor = self._conn.cursor()
        cursor.execute(
            """
                        DELETE
                        FROM
                            langchain_oracle_embedding
                        WHERE
                        collection_id = ( SELECT uuid FROM langchain_oracle_collection WHERE name = :1 )
                     """, [self.collection_name])
        cursor.execute(
            """
                         DELETE
                         FROM
                             langchain_oracle_collection
                         WHERE
                             name = :1
                     """, [self.collection_name])

        cursor.close()
        self._conn.commit()

    def get_by_name(self, name: str) -> Optional[dict]:
        cursor = self._conn.cursor()
        cursor.execute(
            """
                        SELECT
                            name,
                            cmetadata,
                            uuid
                        FROM
                            langchain_oracle_collection
                        WHERE
                            name = :1
                        FETCH FIRST 1 ROWS ONLY
                    """, [name])

        for row in cursor:
            return {"name": row[0], "cmetadata": row[1], "uuid": row[2]}

        return  # type: ignore

    # @get_connection_from_pool
    def get_or_create(
            self,
            name: str,
            cmetadata: Optional[dict] = None,
    ) -> Tuple[dict, bool]:
        """
        Get or create a collection.
        Returns [Collection, bool] where the bool is True if the collection was created.
        """
        created = False
        collection = self.get_collection()
        if collection:
            return collection, created

        cmetadata = json.dumps({}) if cmetadata is None else json.dumps(cmetadata)
        collection_id = uuid.uuid1()
        cursor = self._conn.cursor()
        cursor.execute(
            """
                        INSERT INTO langchain_oracle_collection
                        VALUES (:1, :2, :3)
                     """, [name, cmetadata, str(uuid.uuid1())])
        cursor.close()
        self._conn.commit()

        collection = {"name": name, "cmetadata": cmetadata, "uuid": collection_id}
        created = True
        return collection, created

    # @get_connection_from_pool
    def delete(
            self,
            ids: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> None:
        """Delete vectors by ids or uuids.

        Args:
            ids: List of ids to delete.
        """
        cursor = self._conn.cursor()
        cursor.execute("""
            DELETE FROM langchain_oracle_embedding
            WHERE
                uuid IN ( :1 )
            """, ids)
        self._conn.commit()

    def get_collection(self) -> Optional[dict]:
        return self.get_by_name(self.collection_name)

    @classmethod
    def __from(
            cls,
            texts: List[str],
            embeddings: List[List[float]],
            embedding: Embeddings,
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
            distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
            connection_string: Optional[str] = None,
            pre_delete_collection: bool = False,
            **kwargs: Any,
    ) -> OracleAIVector:
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]

        if not metadatas:
            metadatas = [{} for _ in texts]
        if connection_string is None:
            connection_string = cls.get_connection_string(kwargs)

        store = cls(
            connection_string=connection_string,
            collection_name=collection_name,
            embedding_function=embedding,
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
            **kwargs,
        )

        store.add_embeddings(
            texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids, **kwargs
        )

        return store

    # @get_connection_from_pool
    def add_embeddings(
            self,
            texts: Iterable[str],
            embeddings: List[List[float]],
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> List[str]:
        """Add embeddings to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            embeddings: List of list of embedding vectors.
            metadatas: List of metadatas associated with the texts.
            kwargs: vectorstore specific parameters
        """
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]

        if not metadatas:
            metadatas = [json.dumps({}) for _ in ids]
        else:
            metadatas = [json.dumps(metadata, ensure_ascii=False) for metadata in metadatas]

        collection = self.get_collection()
        if not collection:
            raise ValueError("Collection not found")

        collection_ids = [str(collection["uuid"]) for _ in texts]
        data = list(zip(collection_ids, embeddings, texts, metadatas, ids, ids))

        cursor = self._conn.cursor()
        cursor.setinputsizes(None, oracledb.DB_TYPE_VECTOR, oracledb.DB_TYPE_JSON)
        cursor.executemany(
            """
                        INSERT INTO langchain_oracle_embedding (
                            collection_id,
                            embedding,
                            document,
                            cmetadata,
                            custom_id,
                            uuid
                        ) VALUES (
                            :1,
                            to_vector(:2),
                            :3,
                            :4,
                            :5,
                            :6
                        )        
                     """, data)
        cursor.close()
        self._conn.commit()
        self._conn.close()

        return ids

    def add_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        embeddings = self.embedding_function.embed_documents(list(texts))
        return self.add_embeddings(
            texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids, **kwargs
        )

    def delete_embeddings(
            self,
            source_files: Optional[List[str]] = None,
    ) -> bool:
        """Delete embeddings from specified collection and source files.
            If no source file is not specified, all embeddings related to
            the specified collection will be deleted.

        Args:
            source_files:
                Only delete embeddings related the specified source files. If
                no source file specified, then all embeddings related to the
                collection_name will be deleted.
        """
        if not self.collection_name:
            raise ValueError("Collection not specified")

        collection = self.get_collection()
        if not collection:
            raise ValueError("Collection not found")

        collection_id = str(collection["uuid"])
        cursor = self._conn.cursor()
        if not source_files:
            cursor.execute("""
                DELETE FROM langchain_oracle_embedding
                WHERE COLLECTION_ID = :1
                """, [collection_id])
        else:
            bind_names = ",".join(":" + str(i + 2) for i in range(len(source_files)))
            bind_values = [collection_id] + source_files
            cursor.execute(f"""
                DELETE FROM langchain_oracle_embedding
                WHERE COLLECTION_ID = :1
                    AND json_value(cmetadata, '$.source') IN ( {bind_names} )
                """, bind_values)
        cursor.close()

        self._conn.commit()
        self._conn.close()

        return True

    def similarity_search(
            self,
            query: str,
            k: int = 4,
            filter: Optional[dict] = None,
            **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search with OracleAIVector with distance.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query.
        """
        embedding = self.embedding_function.embed_query(text=query)
        return self.similarity_search_by_vector(
            embedding=embedding,
            k=k,
            filter=filter,
        )

    def similarity_search_with_score(
            self,
            query: str,
            k: int = 4,
            filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query and score for each.
        """
        start_time = datetime.now()  # get the current time
        embedding = self.embedding_function.embed_query(query)
        end_time = datetime.now()  # get the current time
        time_diff = end_time - start_time
        logger.info("######embedding_function.embed_query time:", time_diff)
        docs = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter
        )
        return docs

    @property
    def distance_strategy(self) -> Any:
        if self._distance_strategy == DistanceStrategy.EUCLIDEAN:
            return self.EmbeddingStore.embedding.l2_distance
        elif self._distance_strategy == DistanceStrategy.COSINE:
            return self.EmbeddingStore.embedding.cosine_distance
        elif self._distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
            return self.EmbeddingStore.embedding.max_inner_product
        else:
            raise ValueError(
                f"Got unexpected value for distance: {self._distance_strategy}. "
                f"Should be one of {', '.join([ds.value for ds in DistanceStrategy])}."
            )

    def similarity_search_with_score_by_vector(
            self,
            embedding: List[float],
            k: int = 4,
            filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        results = self.__query_collection(embedding=embedding, k=k, filter=filter)
        return self._results_to_docs_and_scores(results)

    def _results_to_docs_and_scores(self, results: Any) -> List[Tuple[Document, float]]:
        """Return docs and scores from results."""
        # logger.info(results)
        docs = [
            (
                Document(
                    page_content=result["document"],
                    metadata=result["cmetadata"],
                ),
                result["distance"] if self.embedding_function is not None else None,
            )
            for result in results
        ]
        return docs

    # @get_connection_from_pool
    def __query_collection(
            self,
            embedding: List[float],
            k: int = 4,
            filter: Optional[Dict[str, str]] = None,
    ) -> List[Any]:
        """Query the collection."""
        cursor = self._conn.cursor()
        collection = self.get_collection()
        if not collection:
            self.logger.warning("Collection not found")
            return []
        collection_id = collection["uuid"]

        cursor.setinputsizes(oracledb.DB_TYPE_VECTOR)
        # logger.info(embedding[0])
        start_time = datetime.now()  # get the current time
        cursor.execute(
            f"""
                        SELECT 
                            collection_id,
                            document,
                            cmetadata,
                            custom_id,
                            uuid,
                            VECTOR_DISTANCE(embedding, to_vector(:1), {_get_distance_function(self._distance_strategy)}) as distance
                        FROM langchain_oracle_embedding
                        WHERE
                            collection_id = :2
                        ORDER BY distance
                        FETCH FIRST :3 ROWS ONLY
                     """, [embedding, collection_id, k]
        )
        end_time = datetime.now()  # get the current time
        time_diff = end_time - start_time
        logger.info("######Oracle AI Vector Search,VECTOR_DISTANCE(embedding, to_vector(:1)) as distance:", time_diff, " k:",
              k, " collection_id:", collection_id)
        results = cursor.fetchall()
        json_results = []
        for result in results:
            json_results.append(
                {
                    "collection_id": result[0],
                    "document": result[1].read(),
                    "cmetadata": result[2],
                    "custom_id": result[3],
                    "uuid": result[4],
                    "distance": result[5],
                }
            )
        # logger.info(f"results: {results}")
        cursor.close()
        return json_results

    def similarity_search_by_vector(
            self,
            embedding: List[float],
            k: int = 4,
            filter: Optional[dict] = None,
            **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of Documents most similar to the query vector.
        """
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter
        )
        return _results_to_docs(docs_and_scores)

    @classmethod
    def from_texts(
            cls: Type[OracleAIVector],
            texts: List[str],
            embedding: Embeddings,
            metadatas: Optional[List[dict]] = None,
            collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
            distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
            ids: Optional[List[str]] = None,
            pre_delete_collection: bool = False,
            **kwargs: Any,
    ) -> OracleAIVector:
        """
        Return VectorStore initialized from texts and embeddings.
        Postgres connection string is required
        "Either pass it as a parameter
        or set the ORACLE_AI_VECTOR_CONNECTION_STRING environment variable.
        """
        embeddings = embedding.embed_documents(list(texts))

        return cls.__from(
            texts,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
            **kwargs,
        )

    @classmethod
    def from_embeddings(
            cls,
            text_embeddings: List[Tuple[str, List[float]]],
            embedding: Embeddings,
            metadatas: Optional[List[dict]] = None,
            collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
            distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
            ids: Optional[List[str]] = None,
            pre_delete_collection: bool = False,
            **kwargs: Any,
    ) -> OracleAIVector:
        """Construct OracleAIVector wrapper from raw documents and pre-
        generated embeddings.

        Return VectorStore initialized from documents and embeddings.
        Postgres connection string is required
        "Either pass it as a parameter
        or set the ORACLE_AI_VECTOR_CONNECTION_STRING environment variable.

        Example:
            .. code-block:: python

                from mylangchain.vectorstores import OracleAIVector
                from langchain.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                text_embeddings = embeddings.embed_documents(texts)
                text_embedding_pairs = list(zip(texts, text_embeddings))
                faiss = OracleAIVector.from_embeddings(text_embedding_pairs, embeddings)
        """
        texts = [t[0] for t in text_embeddings]
        embeddings = [t[1] for t in text_embeddings]

        return cls.__from(
            texts,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
            **kwargs,
        )

    @classmethod
    def from_existing_index(
            cls: Type[OracleAIVector],
            embedding: Embeddings,
            collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
            distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
            pre_delete_collection: bool = False,
            **kwargs: Any,
    ) -> OracleAIVector:
        """
        Get instance of an existing OracleAIVector store.This method will
        return the instance of the store without inserting any new
        embeddings
        """

        connection_string = cls.get_connection_string(kwargs)

        store = cls(
            connection_string=connection_string,
            collection_name=collection_name,
            embedding_function=embedding,
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
        )

        return store

    @classmethod
    def get_connection_string(cls, kwargs: Dict[str, Any]) -> str:
        connection_string: str = get_from_dict_or_env(
            data=kwargs,
            key="connection_string",
            env_key="ORACLE_AI_VECTOR_CONNECTION_STRING",
        )

        if not connection_string:
            raise ValueError(
                "Oracle AI Vector Search connection string is required"
                "Either pass it as a parameter"
                "or set the ORACLE_AI_VECTOR_CONNECTION_STRING environment variable."
            )

        return connection_string

    @classmethod
    def from_documents(
            cls: Type[OracleAIVector],
            documents: List[Document],
            embedding: Embeddings,
            collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
            distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
            ids: Optional[List[str]] = None,
            pre_delete_collection: bool = False,
            **kwargs: Any,
    ) -> OracleAIVector:
        """
        Return VectorStore initialized from documents and embeddings.
        Postgres connection string is required
        "Either pass it as a parameter
        or set the ORACLE_AI_VECTOR_CONNECTION_STRING environment variable.
        """
        # Oracle Document设置为varchar2(4000)，所以这里设置为：1990
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        connection_string = cls.get_connection_string(kwargs)

        kwargs["connection_string"] = connection_string

        return cls.from_texts(
            texts=texts,
            pre_delete_collection=pre_delete_collection,
            embedding=embedding,
            distance_strategy=distance_strategy,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            **kwargs,
        )

    @classmethod
    def connection_string_from_db_params(
            cls,
            driver: str,
            host: str,
            port: int,
            database: str,
            user: str,
            password: str,
    ) -> str:
        """Return connection string from database parameters."""
        return f"postgresql+{driver}://{user}:{password}@{host}:{port}/{database}"

    def _cosine_relevance_score_fn(self, distance: float) -> float:
        """Normalize the distance to a score on a scale [0, 1]."""
        return 0 if distance > 1.0 else 1.0 - distance

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        The 'correct' relevance function
        may differ depending on a few things, including:
        - the distance / similarity metric used by the VectorStore
        - the scale of your embeddings (OpenAI's are unit normed. Many others are not!)
        - embedding dimensionality
        - etc.
        """
        if self.override_relevance_score_fn is not None:
            return self.override_relevance_score_fn

        # Default strategy is to rely on distance strategy provided
        # in vectorstore constructor
        if self._distance_strategy == DistanceStrategy.COSINE:
            return self._cosine_relevance_score_fn
        elif self._distance_strategy == DistanceStrategy.EUCLIDEAN:
            return self._euclidean_relevance_score_fn
        elif self._distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
            return self._max_inner_product_relevance_score_fn
        else:
            raise ValueError(
                "No supported normalization function"
                f" for distance_strategy of {self._distance_strategy}."
                "Consider providing relevance_score_fn to OracleAIVector constructor."
            )

    def max_marginal_relevance_search_with_score_by_vector(
            self,
            embedding: List[float],
            k: int = 4,
            fetch_k: int = 20,
            lambda_mult: float = 0.5,
            filter: Optional[Dict[str, str]] = None,
            **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs selected using the maximal marginal relevance with score
            to embedding vector.

        Maximal marginal relevance optimizes for similarity to query AND diversity
            among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k (int): Number of Documents to return. Defaults to 4.
            fetch_k (int): Number of Documents to fetch to pass to MMR algorithm.
                Defaults to 20.
            lambda_mult (float): Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
                Defaults to 0.5.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List[Tuple[Document, float]]: List of Documents selected by maximal marginal
                relevance to the query and score for each.
        """
        results = self.__query_collection(embedding=embedding, k=fetch_k, filter=filter)

        embedding_list = [result.get('embedding') for result in results]

        mmr_selected = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            embedding_list,
            k=k,
            lambda_mult=lambda_mult,
        )

        candidates = self._results_to_docs_and_scores(results)

        return [r for i, r in enumerate(candidates) if i in mmr_selected]

    def max_marginal_relevance_search(
            self,
            query: str,
            k: int = 4,
            fetch_k: int = 20,
            lambda_mult: float = 0.5,
            filter: Optional[Dict[str, str]] = None,
            **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
            among selected documents.

        Args:
            query (str): Text to look up documents similar to.
            k (int): Number of Documents to return. Defaults to 4.
            fetch_k (int): Number of Documents to fetch to pass to MMR algorithm.
                Defaults to 20.
            lambda_mult (float): Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
                Defaults to 0.5.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List[Document]: List of Documents selected by maximal marginal relevance.
        """
        embedding = self.embedding_function.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            **kwargs,
        )

    def max_marginal_relevance_search_with_score(
            self,
            query: str,
            k: int = 4,
            fetch_k: int = 20,
            lambda_mult: float = 0.5,
            filter: Optional[dict] = None,
            **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs selected using the maximal marginal relevance with score.

        Maximal marginal relevance optimizes for similarity to query AND diversity
            among selected documents.

        Args:
            query (str): Text to look up documents similar to.
            k (int): Number of Documents to return. Defaults to 4.
            fetch_k (int): Number of Documents to fetch to pass to MMR algorithm.
                Defaults to 20.
            lambda_mult (float): Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
                Defaults to 0.5.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List[Tuple[Document, float]]: List of Documents selected by maximal marginal
                relevance to the query and score for each.
        """
        embedding = self.embedding_function.embed_query(query)
        docs = self.max_marginal_relevance_search_with_score_by_vector(
            embedding=embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )
        return docs

    def max_marginal_relevance_search_by_vector(
            self,
            embedding: List[float],
            k: int = 4,
            fetch_k: int = 20,
            lambda_mult: float = 0.5,
            filter: Optional[Dict[str, str]] = None,
            **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance
            to embedding vector.

        Maximal marginal relevance optimizes for similarity to query AND diversity
            among selected documents.

        Args:
            embedding (str): Text to look up documents similar to.
            k (int): Number of Documents to return. Defaults to 4.
            fetch_k (int): Number of Documents to fetch to pass to MMR algorithm.
                Defaults to 20.
            lambda_mult (float): Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
                Defaults to 0.5.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List[Document]: List of Documents selected by maximal marginal relevance.
        """
        docs_and_scores = self.max_marginal_relevance_search_with_score_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )

        return _results_to_docs(docs_and_scores)

    async def amax_marginal_relevance_search_by_vector(
            self,
            embedding: List[float],
            k: int = 4,
            fetch_k: int = 20,
            lambda_mult: float = 0.5,
            filter: Optional[Dict[str, str]] = None,
            **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance."""

        # This is a temporary workaround to make the similarity search
        # asynchronous. The proper solution is to make the similarity search
        # asynchronous in the vector store implementations.
        func = partial(
            self.max_marginal_relevance_search_by_vector,
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )
        return await asyncio.get_event_loop().run_in_executor(None, func)

    def make_pool_if_not_exists(self):
        global OracleConnectionPool
        if OracleConnectionPool == 1:
            self.pool = oracledb.create_pool(
                # user="vectordemo",
                # password="welcome1",
                dsn=self.connection_string,
                min=1,
                max=3,
                increment=1,
                # wait_timeout=11000,
                # retry_count=1,
                # getmode=oracledb.POOL_GETMODE_TIMEDWAIT
            )
            OracleConnectionPool = self.pool
        else:
            self.pool = OracleConnectionPool


