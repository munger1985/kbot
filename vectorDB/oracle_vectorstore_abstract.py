"""
Description:
 - Implement Oracle VectorStore parent(abstract) class.

 This class(AbstractOracleVS) should not be used directly, instead,
 use its child classes: OracleAdbVS for ADB and OracleBaseDbVS for
 Oracle Base Database.


History:
 - 2024/09/07 by Hysun (hysun.he@oracle.com): Created
"""

from __future__ import annotations

import functools
import json
import uuid
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)
import oracledb
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langchain_community.vectorstores.utils import (
    DistanceStrategy,
    maximal_marginal_relevance,
)
from loguru import logger

_EMBEDDING_TABLE_NAME = "kbot_oracle_embeddings"


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


# Define a type variable that can be any kind of function
T = TypeVar("T", bound=Callable[..., Any])


def _handle_exceptions(func: T) -> T:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except RuntimeError as db_err:
            # Handle a known type of error (e.g., DB-related) specifically
            logger.exception("DB-related error occurred.")
            raise RuntimeError(
                "Failed due to a DB issue: {}".format(db_err)
            ) from db_err
        except ValueError as val_err:
            # Handle another known type of error specifically
            logger.exception("Validation error.")
            raise ValueError("Validation failed: {}".format(val_err)) from val_err
        except Exception as e:
            # Generic handler for all other exceptions
            logger.exception("An unexpected error occurred: {}".format(e))
            raise RuntimeError("Unexpected error: {}".format(e)) from e

    return cast(T, wrapper)


class AbstractOracleVS(VectorStore):
    """`AbstractOracleVS` vector store.

    To use, you should have both:
    - the ``oracledb`` python package installed
    """

    def __init__(
        self,
        collection_name: str,
        embedding_function: Union[
            Callable[[str], List[float]],
            Embeddings,
        ],
        pre_delete_collection: bool = False,
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
        params: Optional[Dict[str, Any]] = None,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
    ):
        try:
            """Initialize with necessary components."""
            if not isinstance(embedding_function, Embeddings):
                logger.warning(
                    "`embedding_function` is expected to be an Embeddings "
                    "object, support "
                    "for passing in a function will soon be removed."
                )
            self.collection_name = collection_name
            self.embedding_function = embedding_function
            self.distance_strategy = distance_strategy
            self.params = params
            self.table_name = _EMBEDDING_TABLE_NAME
            self.pre_delete_collection = pre_delete_collection
            self.override_relevance_score_fn = relevance_score_fn

            self.create_or_clean_table()
        except oracledb.DatabaseError as db_err:
            logger.exception(f"Database error occurred while create table: {db_err}")
            raise RuntimeError(
                "Failed to create table due to a database error."
            ) from db_err
        except ValueError as val_err:
            logger.exception(f"Validation error: {val_err}")
            raise RuntimeError(
                "Failed to create table due to a validation error."
            ) from val_err
        except Exception as ex:
            logger.exception("An unexpected error occurred while creating the index.")
            raise RuntimeError(
                "Failed to create table due to an unexpected error."
            ) from ex

    def connect(self) -> oracledb.Connection:
        """Sub class must override this function to create and return a
        valid database connection.

        Returns:
            oracledb.Connection: Oracle database connection object
        """
        raise NotImplementedError

    def table_exists(self) -> bool:
        with self.connect() as client:
            with client.cursor() as cursor:
                try:
                    cursor.execute(f"SELECT 1 FROM {self.table_name} where rownum < 2")
                    results = cursor.fetchall()
                    logger.info(f"### Check table exists? {results}")
                    return True
                except oracledb.DatabaseError as ex:
                    err_obj = ex.args
                    if err_obj[0].code == 942:
                        return False
                    raise

    @_handle_exceptions
    def create_or_clean_table(self) -> None:
        if not self.table_exists():
            logger.info(f"### create embedding table: {self.table_name}")
            with self.connect() as client:
                with client.cursor() as cursor:
                    ddl = f"""
                        create table {self.table_name} (
                            id varchar2(256) NOT NULL PRIMARY KEY,
                            collection_name varchar2(256),
                            embedding vector(*, FLOAT32),
                            document CLOB,
                            metadata JSON,
                            chunk_category varchar2(256)
                        )
                    """
                    results: Any = cursor.execute(ddl)
                    logger.info(f"### create table results: {results}")
            logger.info("### Table created successfully...")
        elif self.pre_delete_collection:
            logger.info(f"### Delete existing data: {self.collection_name}")
            with self.connect() as client:
                with client.cursor() as cursor:
                    sql = f"delete from {self.table_name} WHERE collection_name = '{self.collection_name}'"
                    cursor.execute(sql)
                    client.commit()
            logger.info(f"### Data deleted: {self.collection_name}")
        else:
            logger.info(f"### create_or_clean_table: No action needed")

    @property
    def embeddings(self) -> Optional[Embeddings]:
        """
        A property that returns an Embeddings instance embedding_function
        is an instance of Embeddings, otherwise returns None.

        Returns:
            Optional[Embeddings]: The embedding function if it's an instance of
            Embeddings, otherwise None.
        """
        return (
            self.embedding_function
            if isinstance(self.embedding_function, Embeddings)
            else None
        )

    def _embed_documents(self, texts: List[str]) -> List[List[float]]:
        if isinstance(self.embedding_function, Embeddings):
            return self.embedding_function.embed_documents(texts)
        elif callable(self.embedding_function):
            return [self.embedding_function(text) for text in texts]
        else:
            raise TypeError(
                "The embedding_function is neither Embeddings nor callable."
            )

    def _embed_query(self, text: str) -> List[float]:
        if isinstance(self.embedding_function, Embeddings):
            return self.embedding_function.embed_query(text)
        else:
            return self.embedding_function(text)

    @_handle_exceptions
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add more texts to the vectorstore index.
        Args:
          texts: Iterable of strings to add to the vectorstore.
          metadatas: Optional list of metadatas associated with the texts.
          ids: Optional list of ids for the texts that are being added to
          the vector store.
          kwargs: vectorstore specific parameters
        """

        texts = list(texts)
        if ids:
            # If ids are provided, hash them to maintain consistency
            processed_ids = [_id for _id in ids]
        elif metadatas and all("id" in metadata for metadata in metadatas):
            # If no ids are provided but metadatas with ids are, generate
            # ids from metadatas
            processed_ids = [metadata["id"] for metadata in metadatas]
        else:
            # Generate new ids if none are provided
            processed_ids = [str(uuid.uuid4()) for _ in texts]

        embeddings = self._embed_documents(texts)

        if not metadatas:
            metadatas = [{} for _ in ids]

        docs = [
            (
                id_,
                self.collection_name,
                metadata,
                # json.dumps(metadata, ensure_ascii=False),
                embedding,
                metadata.get("chunk_category", "SOURCE_DOCS") or "SOURCE_DOCS",
                text,
            )
            for id_, text, metadata, embedding in zip(
                processed_ids, texts, metadatas, embeddings
            )
        ]

        with self.connect() as client:
            with client.cursor() as cursor:
                cursor.setinputsizes(
                    None,
                    None,
                    oracledb.DB_TYPE_JSON,
                    oracledb.DB_TYPE_VECTOR,
                    None,
                    oracledb.DB_TYPE_CLOB,
                )

                cursor.executemany(
                    f"INSERT INTO {self.table_name} (id, collection_name,  metadata, embedding, chunk_category, document) VALUES (:1, :2, :3, :4, :5, :6)",
                    docs,
                )
                client.commit()
        return processed_ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query."""
        if isinstance(self.embedding_function, Embeddings):
            embedding = self.embedding_function.embed_query(query)
        documents = self.similarity_search_by_vector(
            embedding=embedding, k=k, filter=filter, **kwargs
        )
        return documents

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        docs_and_scores = self.similarity_search_by_vector_with_relevance_scores(
            embedding=embedding, k=k, filter=filter, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query."""
        if isinstance(self.embedding_function, Embeddings):
            embedding = self.embedding_function.embed_query(query)
        docs_and_scores = self.similarity_search_by_vector_with_relevance_scores(
            embedding=embedding, k=k, filter=filter, **kwargs
        )
        return docs_and_scores

    @_handle_exceptions
    def similarity_search_by_vector_with_relevance_scores(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        docs_and_scores = []
        embedding_str = json.dumps(embedding)
        if kwargs.get("summary_flag") == "Y":
            logger.info("### Search chunk summary vectors")

            query = f"""
            SELECT t2.id,
                t2.document,
                t2.metadata,
                t1.distance
            FROM (
                SELECT
                    t.metadata.src_chunk_id as src_chunk_id,
                    vector_distance(embedding, to_vector(:1), {_get_distance_function(
                        self.distance_strategy)}) as distance
                FROM {self.table_name} t
                WHERE collection_name = '{self.collection_name}'
                    and chunk_category like 'SUMMARY_%'
                ORDER BY distance
                FETCH FIRST {k} ROWS ONLY
            ) t1, {self.table_name} t2
            WHERE t1.src_chunk_id = t2.metadata.chunk_id
            
            union all

            select * from (
                SELECT id,
                    document,
                    metadata,
                    vector_distance(embedding, to_vector(:2), {_get_distance_function(
                        self.distance_strategy)}) as distance
                FROM {self.table_name} t1
                WHERE collection_name = '{self.collection_name}'
                    AND ( t1.chunk_category is null OR (
                        t1.chunk_category like 'SOURCE_%'
                        and not exists (
                            select 1 from kbot_oracle_embeddings t2 
                            where t1.metadata.chunk_id = t2.metadata.src_chunk_id
                        )
                    ))
                ORDER BY distance
                FETCH FIRST {k} ROWS ONLY
            )

            ORDER BY distance
            FETCH FIRST {k} ROWS ONLY
            """
            bind_params = [embedding_str, embedding_str]
        else:
            logger.info("### Search chunk vectors")

            query = f"""
            SELECT id,
                document,
                metadata,
                vector_distance(embedding, to_vector(:3), {_get_distance_function(
                    self.distance_strategy)}) as distance
            FROM {self.table_name}
            WHERE collection_name = '{self.collection_name}'
                and (chunk_category is null or chunk_category like 'SOURCE_%')
            ORDER BY distance
            FETCH FIRST {k} ROWS ONLY
            """
            bind_params = [embedding_str]

        # Execute the query
        with self.connect() as client:
            with client.cursor() as cursor:
                cursor.execute(query, bind_params)
                results = cursor.fetchall()

                # Filter results if filter is provided
                for result in results:
                    metadata = result[2]
                    # metadata = json.loads(result[2]) if result[2] else None

                    # Apply filtering based on the 'filter' dictionary
                    if filter:
                        if all(
                            metadata.get(key) in value for key, value in filter.items()
                        ):
                            doc = Document(
                                page_content=str(result[1]),
                                metadata=metadata,
                            )
                            distance = result[3]
                            docs_and_scores.append((doc, distance))
                    else:
                        doc = Document(
                            page_content=str(result[1]),
                            metadata=metadata,
                        )
                        distance = result[3]
                        docs_and_scores.append((doc, distance))

        return docs_and_scores

    @_handle_exceptions
    def similarity_search_by_vector_returning_embeddings(
        self,
        embedding: List[float],
        k: int,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float, np.ndarray[np.float32, Any]]]:
        documents = []
        query = f"""
        SELECT id,
          document,
          metadata,
          vector_distance(embedding, to_vector(:1), {_get_distance_function(
            self.distance_strategy)}) as distance,
          embedding
        FROM {self.table_name}
        WHERE collection_name = '{self.collection_name}'
        ORDER BY distance
        FETCH FIRST {k} ROWS ONLY
        """

        # Execute the query
        with self.connect() as client:
            with client.cursor() as cursor:
                cursor.execute(query, [json.dumps(embedding)])
                results = cursor.fetchall()

                for result in results:
                    page_content_str = str(result[1])
                    metadata = result[2]

                    # Apply filter if provided and matches; otherwise, add all
                    # documents
                    if not filter or all(
                        metadata.get(key) in value for key, value in filter.items()
                    ):
                        document = Document(
                            page_content=page_content_str, metadata=metadata
                        )
                        distance = result[3]

                        # Assuming result[4] is already in the correct format;
                        # adjust if necessary
                        current_embedding = (
                            np.array(result[4], dtype=np.float32)
                            if result[4]
                            else np.empty(0, dtype=np.float32)
                        )

                        documents.append((document, distance, current_embedding))
        return documents  # type: ignore

    @_handle_exceptions
    def max_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: List[float],
        *,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Return docs and their similarity scores selected using the
        maximal marginal
            relevance.

        Maximal marginal relevance optimizes for similarity to query AND
        diversity
        among selected documents.

        Args:
          self: An instance of the class
          embedding: Embedding to look up documents similar to.
          k: Number of Documents to return. Defaults to 4.
          fetch_k: Number of Documents to fetch before filtering to
                   pass to MMR algorithm.
          filter: (Optional[Dict[str, str]]): Filter by metadata. Defaults
          to None.
          lambda_mult: Number between 0 and 1 that determines the degree
                       of diversity among the results with 0 corresponding
                       to maximum diversity and 1 to minimum diversity.
                       Defaults to 0.5.
        Returns:
            List of Documents and similarity scores selected by maximal
            marginal
            relevance and score for each.
        """

        # Fetch documents and their scores
        docs_scores_embeddings = self.similarity_search_by_vector_returning_embeddings(
            embedding, fetch_k, filter=filter
        )
        # Assuming documents_with_scores is a list of tuples (Document, score)

        # If you need to split documents and scores for processing (e.g.,
        # for MMR calculation)
        documents, scores, embeddings = (
            zip(*docs_scores_embeddings) if docs_scores_embeddings else ([], [], [])
        )

        # Assume maximal_marginal_relevance method accepts embeddings and
        # scores, and returns indices of selected docs
        mmr_selected_indices = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            list(embeddings),
            k=k,
            lambda_mult=lambda_mult,
        )

        # Filter documents based on MMR-selected indices and map scores
        mmr_selected_documents_with_scores = [
            (documents[i], scores[i]) for i in mmr_selected_indices
        ]

        return mmr_selected_documents_with_scores

    @_handle_exceptions
    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND
        diversity
        among selected documents.

        Args:
          self: An instance of the class
          embedding: Embedding to look up documents similar to.
          k: Number of Documents to return. Defaults to 4.
          fetch_k: Number of Documents to fetch to pass to MMR algorithm.
          lambda_mult: Number between 0 and 1 that determines the degree
                       of diversity among the results with 0 corresponding
                       to maximum diversity and 1 to minimum diversity.
                       Defaults to 0.5.
          filter: Optional[Dict[str, Any]]
          **kwargs: Any
        Returns:
          List of Documents selected by maximal marginal relevance.
        """
        docs_and_scores = self.max_marginal_relevance_search_with_score_by_vector(
            embedding=embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
        )
        return [doc for doc, _ in docs_and_scores]

    @_handle_exceptions
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND
        diversity
        among selected documents.

        Args:
          self: An instance of the class
          query: Text to look up documents similar to.
          k: Number of Documents to return. Defaults to 4.
          fetch_k: Number of Documents to fetch to pass to MMR algorithm.
          lambda_mult: Number between 0 and 1 that determines the degree
                       of diversity among the results with 0 corresponding
                       to maximum diversity and 1 to minimum diversity.
                       Defaults to 0.5.
          filter: Optional[Dict[str, Any]]
          **kwargs
        Returns:
          List of Documents selected by maximal marginal relevance.

        `max_marginal_relevance_search` requires that `query` returns matched
        embeddings alongside the match documents.
        """
        embedding = self._embed_query(query)
        documents = self.max_marginal_relevance_search_by_vector(
            embedding=embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )
        return documents

    @classmethod
    def from_documents(
        cls: Type[AbstractOracleVS],
        collection_name: str,
        documents: List[Document],
        embedding: Embeddings,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> AbstractOracleVS:
        """Return VectorStore initialized from documents and embeddings.

        Args:
            documents: List of Documents to add to the vectorstore.
            embedding: Embedding function to use.
            **kwargs: Additional keyword arguments.

        Returns:
            VectorStore: VectorStore initialized from documents and embeddings.
        """
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        return cls.from_texts(
            collection_name=collection_name,
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            pre_delete_collection=pre_delete_collection,
            **kwargs,
        )

    @classmethod
    @_handle_exceptions
    def from_texts(
        cls: Type[AbstractOracleVS],
        collection_name: str,
        texts: Iterable[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> AbstractOracleVS:
        """Return VectorStore initialized from texts and embeddings."""
        params = kwargs.get("params", {})

        distance_strategy = cast(
            DistanceStrategy,
            kwargs.get("distance_strategy", DistanceStrategy.COSINE),
        )
        if not isinstance(distance_strategy, DistanceStrategy):
            raise TypeError(
                f"Expected DistanceStrategy got " f"{type(distance_strategy).__name__} "
            )

        vss = cls(
            collection_name=collection_name,
            embedding_function=embedding,
            distance_strategy=distance_strategy,
            pre_delete_collection=pre_delete_collection,
            params=params,
        )
        vss.add_texts(texts=list(texts), metadatas=metadatas)
        return vss

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

        if not source_files:
            with self.connect() as client:
                with client.cursor() as cursor:
                    cursor.execute(
                        f"delete from {self.table_name} WHERE collection_name = '{self.collection_name}'"
                    )
                    logger.info(f"### delete_embeddings result: {cursor.rowcount}")
                    client.commit()
        else:
            bind_names = ",".join(":" + str(i + 2) for i in range(len(source_files)))
            bind_values = [self.collection_name] + source_files
            with self.connect() as client:
                with client.cursor() as cursor:
                    cursor.execute(
                        f"""
                        DELETE FROM {self.table_name}
                        WHERE collection_name = :1
                            AND json_value(metadata, '$.source') IN ( {bind_names} )
                        """,
                        bind_values,
                    )
                    logger.info(f"### delete_embeddings result: {cursor.rowcount}")
                    client.commit()
        return True

    def _cosine_relevance_score_fn(self, distance: float) -> float:
        """Normalize the distance to a score on a scale [0, 1]."""
        return 0 if distance > 1.0 else 1.0 - distance

    def _constant_relevance_score_fn(self, distance: float) -> float:
        """Output directly"""
        return distance

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
        if self.distance_strategy == DistanceStrategy.COSINE:
            return self._cosine_relevance_score_fn
        elif self.distance_strategy == DistanceStrategy.EUCLIDEAN:
            return self._euclidean_relevance_score_fn
        else:
            logger.warning(
                "No supported normalization function"
                f" for distance_strategy of {self._distance_strategy}."
            )
            return self._constant_relevance_score_fn
