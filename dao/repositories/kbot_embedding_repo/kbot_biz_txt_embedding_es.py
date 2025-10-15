import json
from typing import Sequence, Optional
from elasticsearch import AsyncElasticsearch
from loguru import logger
from dao.entities.kbot_biz_txt_embedding import KbotBizTxtEmbedding
from dao.repositories.kbot_md_db_conf_repo import KbotMdDbConfRepository
from core.dictionary import ChunkType
from dao.repositories.kbot_biz_txt_embedding_interface import IEmbeddingRepository


class ElasticsearchEmbeddingRepository(IEmbeddingRepository):
    """Elasticsearch 9.1.5 实现 - 与Oracle接口完全兼容"""
    
    def __init__(self, kb_id: int):
        self.kb_id = kb_id
        self.es_client: Optional[AsyncElasticsearch] = None
        self.index_name = f"kbot_biz_txt_embedding_{kb_id}"
        self.db_conf = None

    async def initialize(self) -> bool:
        """初始化ES连接"""
        try:
            db_repo = KbotMdDbConfRepository()
            self.db_conf = await db_repo.get_by_kbid(self.kb_id)
            if self.db_conf is None:
                logger.error(f"未找到知识库 {self.kb_id} 的数据库配置")
                return False
            
            es_config = self.db_conf.db_conn_str
            if es_config is None:
                logger.error(f"知识库 {self.kb_id} 的ES连接配置为空")
                return False
            
            # 创建ES客户端
            hosts = [f"{es_config.get('host', 'localhost')}:{es_config.get('port', 9200)}"]
            http_auth = None
            if es_config.get("user") and es_config.get("password"):
                http_auth = (es_config.get("user"), es_config.get("password"))
            
            self.es_client = AsyncElasticsearch(
                hosts=hosts,
                http_auth=http_auth,
                scheme=es_config.get("scheme", "http"), # type: ignore
                verify_certs=es_config.get("verify_certs", False),
                ssl_show_warn=es_config.get("ssl_show_warn", False)
            )
            
            # 检查连接
            if not await self.es_client.ping():
                logger.error("ES连接测试失败")
                return False
            
            # 检查索引是否存在，不存在则创建
            if not await self.es_client.indices.exists(index=self.index_name):
                await self._create_index()
            
            logger.info(f"ES存储库初始化成功，索引: {self.index_name}")
            return True
            
        except Exception as e:
            logger.error(f"初始化ES连接失败: {e}")
            return False

    async def _create_index(self):
        """创建ES索引和映射"""
        mapping = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 1,
                "analysis": {
                    "analyzer": {
                        "chinese_analyzer": {
                            "type": "custom",
                            "tokenizer": "ik_max_word"
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "embed_id": {"type": "keyword"},
                    "kb_id": {"type": "integer"},
                    "file_id": {"type": "keyword"},
                    "security_level": {"type": "integer"},
                    "chunk_metadata": {
                        "type": "object",
                        "properties": {
                            "chunk_type": {"type": "keyword"},
                            "source_embed_id": {"type": "keyword"},
                            "chunk_number": {"type": "integer"}
                        }
                    },
                    "biz_metadata": {
                        "type": "object",
                        "properties": {
                            "tags": {"type": "keyword"}
                        }
                    },
                    "chunk_doc": {
                        "type": "text",
                        "analyzer": "chinese_analyzer",
                        "search_analyzer": "ik_smart"
                    },
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 1536,  # 根据你的向量维度调整
                        "index": True,
                        "similarity": "cosine"
                    },
                    "status": {"type": "integer"}
                }
            }
        }
        await self.es_client.indices.create(index=self.index_name, body=mapping) # type: ignore
        logger.info(f"创建ES索引: {self.index_name}")

    async def create(self, kb_id: int, embeddings: list[KbotBizTxtEmbedding]) -> bool:
        """批量创建嵌入记录 - 与Oracle接口完全一致"""
        if self.es_client is None or not embeddings:
            return False
        
        try:
            operations = []
            for embedding in embeddings:
                # 构建文档，字段与Oracle表结构对应
                doc = {
                    "embed_id": embedding.embed_id,
                    "kb_id": kb_id,
                    "file_id": embedding.file_id,
                    "security_level": embedding.security_level,
                    "chunk_metadata": embedding.chunk_metadata or {},
                    "biz_metadata": embedding.biz_metadata or {},
                    "chunk_doc": embedding.chunk_doc,
                    "embedding": embedding.embedding,  # 直接使用向量列表
                    "status": 1  # 默认状态
                }
                
                operations.append({"index": {"_index": self.index_name, "_id": embedding.embed_id}})
                operations.append(doc)
            
            # 批量插入
            response = await self.es_client.bulk(operations=operations, refresh=True)
            
            if response["errors"]:
                errors = []
                for item in response["items"]:
                    if "error" in item["index"]:
                        errors.append(item["index"]["error"])
                logger.error(f"ES批量插入失败: {errors}")
                return False
            
            logger.info(f"ES成功批量插入 {len(embeddings)} 条记录")
            return True
            
        except Exception as e:
            logger.error(f"ES批量插入失败: {e}")
            return False

    async def delete_by_file_ids(self, kb_id: int, file_ids: list[str]) -> int:
        """根据文件ID删除嵌入记录 - 与Oracle接口完全一致"""
        if self.es_client is None or not file_ids:
            return 0
        
        try:
            # 构建删除查询
            query = {
                "bool": {
                    "must": [
                        {"term": {"kb_id": kb_id}},
                        {"terms": {"file_id": file_ids}}
                    ]
                }
            }
            
            response = await self.es_client.delete_by_query(
                index=self.index_name,
                body={"query": query},
                refresh=True
            )
            
            deleted_count = response["deleted"]
            logger.info(f"ES成功删除 {deleted_count} 条记录，文件IDs: {file_ids}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"ES按文件ID删除失败: {e}")
            return 0

    async def get_similar_embeddings(self,
                                   kb_id: int,
                                   query_vec: str,
                                   security: int,
                                   similarity_threshold: Optional[float] = 0.8,
                                   top_k: Optional[int] = 10,
                                   is_summary_search: bool = False,
                                   tags: Optional[list[str]] = None) -> Sequence:
        """向量相似度搜索 - 与Oracle接口完全一致"""
        if self.es_client is None:
            return []
        
        try:
            # 解析查询向量（Oracle传入的是字符串格式）
            if isinstance(query_vec, str):
                query_vector = json.loads(query_vec)
            else:
                query_vector = query_vec
            
            # 构建过滤条件
            filter_conditions = [
                {"term": {"kb_id": kb_id}},
                {"range": {"security_level": {"lte": security}}},
                {"term": {"chunk_metadata.chunk_type": ChunkType.SUMMARY.value if is_summary_search else ChunkType.TEXT.value}}
            ]
            
            # 标签过滤
            if tags and len(tags) > 0:
                tag_conditions = [{"term": {"biz_metadata.tags": tag}} for tag in tags]
                filter_conditions.append({"bool": {"should": tag_conditions, "minimum_should_match": 1}})
            
            # 构建向量搜索查询
            query = {
                "script_score": {
                    "query": {
                        "bool": {
                            "filter": filter_conditions
                        }
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {
                            "query_vector": query_vector
                        }
                    }
                }
            }
            
            response = await self.es_client.search(
                index=self.index_name,
                body={
                    "query": query,
                    "size": top_k or 10,
                    "_source": ["file_id", "chunk_doc", "chunk_metadata"],
                    "min_score": (similarity_threshold or 0.8) + 1.0,  # ES的cosineSimilarity返回0-2
                    "sort": [{"_score": "desc"}]
                }
            )
            
            # 转换结果格式，与Oracle返回格式完全一致
            results = []
            for hit in response["hits"]["hits"]:
                similarity = hit["_score"] - 1.0  # 转换回0-1范围
                results.append((
                    hit["_source"]["file_id"],
                    hit["_source"]["chunk_doc"],
                    hit["_source"]["chunk_metadata"],
                    similarity
                ))
            
            logger.info(f"ES向量搜索返回 {len(results)} 条结果")
            return results
            
        except Exception as e:
            logger.error(f"ES向量搜索失败: {e}")
            return []

    async def full_text_search(self,
                             kb_id: int,
                             keyword: str,
                             security: int,
                             top_k: Optional[int] = 10,
                             similarity_threshold: Optional[float] = 0.8,
                             tags: Optional[list[str]] = None) -> Sequence:
        """全文检索 - 与Oracle接口完全一致"""
        if self.es_client is None:
            return []
        
        try:
            # 构建查询条件
            must_conditions = [
                {"term": {"kb_id": kb_id}},
                {"range": {"security_level": {"lte": security}}},
                {
                    "match": {
                        "chunk_doc": {
                            "query": keyword,
                            "operator": "or"
                        }
                    }
                }
            ]
            
            # 标签过滤
            if tags and len(tags) > 0:
                tag_conditions = [{"term": {"biz_metadata.tags": tag}} for tag in tags]
                must_conditions.append({"bool": {"should": tag_conditions, "minimum_should_match": 1}})
            
            query = {"bool": {"must": must_conditions}}
            
            response = await self.es_client.search(
                index=self.index_name,
                body={
                    "query": query,
                    "size": top_k or 10,
                    "_source": ["file_id", "chunk_doc", "chunk_metadata"],
                    "min_score": similarity_threshold or 0.0,
                    "sort": [{"_score": "desc"}]
                }
            )
            
            # 转换结果格式，与Oracle返回格式完全一致
            results = []
            for hit in response["hits"]["hits"]:
                results.append((
                    hit["_source"]["file_id"],
                    hit["_source"]["chunk_doc"],
                    hit["_source"]["chunk_metadata"],
                    hit["_score"]
                ))
            
            logger.info(f"ES全文检索返回 {len(results)} 条结果")
            return results
            
        except Exception as e:
            logger.error(f"ES全文检索失败: {e}")
            return []

    async def update_chunk(self,
                          embed_id: str,
                          new_chunk: str,
                          new_embedding: list[float]) -> bool:
        """更新块内容和嵌入向量 - 与Oracle接口完全一致"""
        if self.es_client is None:
            return False
        
        try:
            update_body = {
                "doc": {
                    "chunk_doc": new_chunk,
                    "embedding": new_embedding
                }
            }
            
            response = await self.es_client.update(
                index=self.index_name,
                id=embed_id,
                body=update_body,
                refresh=True
            )
            
            if response["result"] in ["updated", "noop"]:
                logger.info(f"ES成功更新块: {embed_id}")
                return True
            else:
                logger.error(f"ES更新块失败: {response['result']}")
                return False
                
        except Exception as e:
            logger.error(f"ES更新块失败: {e}")
            return False

    async def get_summary_id_by_chunk_id(self, file_id: str, chunk_id: str) -> Optional[str]:
        """根据块ID获取摘要ID - 与Oracle接口完全一致"""
        if self.es_client is None:
            return None
        
        try:
            query = {
                "bool": {
                    "must": [
                        {"term": {"file_id": file_id}},
                        {"term": {"chunk_metadata.chunk_type": ChunkType.SUMMARY.value}},
                        {"term": {"chunk_metadata.source_embed_id": chunk_id}}
                    ]
                }
            }
            
            response = await self.es_client.search(
                index=self.index_name,
                body={
                    "query": query,
                    "size": 1,
                    "_source": ["embed_id"]
                }
            )
            
            if response["hits"]["hits"]:
                embed_id = response["hits"]["hits"][0]["_source"]["embed_id"]
                logger.info(f"ES找到摘要ID: {embed_id} for chunk: {chunk_id}")
                return embed_id
            
            logger.info(f"ES未找到chunk {chunk_id} 对应的摘要")
            return None
            
        except Exception as e:
            logger.error(f"ES获取摘要ID失败: {e}")
            return None

    async def delete_by_embed_ids(self, embed_ids: list[str]) -> int:
        """根据嵌入ID删除记录 - 与Oracle接口完全一致"""
        if self.es_client is None or not embed_ids:
            return 0
        
        try:
            # 构建删除查询
            query = {
                "terms": {"embed_id": embed_ids}
            }
            
            response = await self.es_client.delete_by_query(
                index=self.index_name,
                body={"query": query},
                refresh=True
            )
            
            deleted_count = response["deleted"]
            logger.info(f"ES成功删除 {deleted_count} 条记录，嵌入IDs: {embed_ids}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"ES按嵌入ID删除失败: {e}")
            return 0

    async def update_status_by_chunk_id(self, chunk_id: str, status: int) -> int:
        """更新块状态 - 与Oracle接口完全一致"""
        if self.es_client is None:
            return 0
        
        try:
            # 构建更新查询
            query = {
                "term": {"embed_id": chunk_id}
            }
            
            update_body = {
                "script": {
                    "source": "ctx._source.status = params.status",
                    "params": {"status": status}
                }
            }
            
            response = await self.es_client.update_by_query(
                index=self.index_name,
                body={
                    "query": query,
                    "script": update_body["script"]
                },
                refresh=True
            )
            
            updated_count = response["updated"]
            logger.info(f"ES成功更新 {updated_count} 条记录的状态，chunk_id: {chunk_id}, status: {status}")
            return updated_count
            
        except Exception as e:
            logger.error(f"ES更新状态失败: {e}")
            return 0

    async def close(self):
        """关闭ES连接"""
        if self.es_client:
            await self.es_client.close()
            logger.info("ES连接已关闭")