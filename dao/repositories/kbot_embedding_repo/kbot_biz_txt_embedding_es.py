import json
from typing import Sequence
from elasticsearch import AsyncElasticsearch
from loguru import logger
from dao.entities.kbot_biz_txt_embedding import KbotBizTxtEmbedding
from core.dictionary import ChunkType
from dao.repositories.kbot_biz_txt_embedding_interface import IEmbeddingRepository
from core.database.vec_elasticsearch import es_client_manager

class ElasticsearchEmbeddingRepository(IEmbeddingRepository):
    """Elasticsearch 9.1.5 实现 - 与Oracle接口完全兼容"""
    
    def __init__(self, kb_id: int):
        self.kb_id = kb_id
        self.es_client: AsyncElasticsearch | None = None
        self.index_name = f"kbot_biz_txt_embedding_{kb_id}"


    async def initialize(self, connstr: dict) -> bool:
        """初始化ES连接"""
        try:
            if connstr is not None:
                self.connstr = connstr
                
                # 通过单例管理器获取ES客户端
                self.es_client = await es_client_manager.get_client(connstr)
                if self.es_client is None:
                    logger.error("获取ES客户端失败")
                    return False
                
                # 检查索引是否存在，不存在则创建
                if not await self.es_client.indices.exists(index=self.index_name):
                    await self._create_index()
                
                logger.info(f"ES存储库初始化成功，索引: {self.index_name}")
                return True
            else:
                logger.error("ES连接参数为空")
                return False
            
        except Exception as e:
            logger.exception(f"初始化ES连接失败: {e}")
            return False


    async def _create_index(self):
        """创建ES索引和映射 - 使用正确的向量维度"""
        mapping = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0  # 单节点环境可以设为0
            },
            "mappings": {
                "properties": {
                    "embed_id": {"type": "keyword"},
                    "kb_id": {"type": "integer"},
                    "file_id": {"type": "keyword"},
                    "security_level": {"type": "integer"},
                    "chunk_metadata": {"type": "object"},
                    "biz_metadata": {"type": "object"},
                    "chunk_doc": {"type": "text"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 1024,  # 根据实际向量维度调整为1024
                        "index": True,
                        "similarity": "cosine"
                    },
                    "status": {"type": "integer"}
                }
            }
        }
        
        try:
            await self.es_client.indices.create(index=self.index_name, body=mapping) # type: ignore
            logger.info(f"创建ES索引成功: {self.index_name}")
        except Exception as e:
            logger.error(f"创建ES索引失败: {e}")
            raise

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
                               similarity_threshold: float = 0.8,
                               search_top_k: int = 10,
                               is_summary_search: bool = False,
                               tags: list[str] = []
                               ) -> Sequence:
        """向量相似度搜索"""
        if self.es_client is None:
            return []
        
        try:
            # 解析查询向量
            if isinstance(query_vec, str):
                query_vector = json.loads(query_vec)
            else:
                query_vector = query_vec
            
            # 构建基础过滤条件
            filter_conditions = [
                {"term": {"kb_id": kb_id}},
                {"range": {"security_level": {"lte": security}}},
                {"term": {"chunk_metadata.chunk_type": ChunkType.SUMMARY.value if is_summary_search else ChunkType.TEXT.value}}
            ]
            
            # 只有当tags非空时才添加条件
            if tags and len(tags) > 0:
                tag_conditions = [{"term": {"biz_metadata.tags": tag}} for tag in tags]
                filter_conditions.append({"bool": {"should": tag_conditions, "minimum_should_match": 1}})
            # 如果tags为空或None，不添加任何tag条件
            
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
                    "size": search_top_k or 10,
                    "_source": ["file_id", "chunk_doc", "chunk_metadata"],
                    "min_score": (similarity_threshold or 0.8) + 1.0,
                    "sort": [{"_score": "desc"}]
                }
            )
            
            # 转换结果格式
            results = []
            for hit in response["hits"]["hits"]:
                similarity = hit["_score"] - 1.0
                results.append((
                    hit["_source"]["file_id"],
                    hit["_source"]["chunk_doc"],
                    hit["_source"]["chunk_metadata"],
                    similarity
                ))
            
            logger.info(f"ES向量搜索返回 {len(results)} 条结果，tags: {tags}")
            return results
            
        except Exception as e:
            logger.error(f"ES向量搜索失败: {e}")
            return []

    async def full_text_search(self,
                            kb_id: int,
                            keyword: str,
                            security: int,
                            search_top_k: int = 10,
                            similarity_threshold: float = 0.8,
                            tags: list[str] = []
                            ) -> Sequence:
        """全文检索"""
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
            
            # 修复tags处理：只有当tags非空时才添加条件
            if tags and len(tags) > 0:
                tag_conditions = [{"term": {"biz_metadata.tags": tag}} for tag in tags]
                must_conditions.append({"bool": {"should": tag_conditions, "minimum_should_match": 1}})
            # 如果tags为空或None，不添加任何tag条件
            
            query = {"bool": {"must": must_conditions}}
            
            response = await self.es_client.search(
                index=self.index_name,
                body={
                    "query": query,
                    "size": search_top_k or 10,
                    "_source": ["file_id", "chunk_doc", "chunk_metadata"],
                    "min_score": similarity_threshold or 0.0,
                    "sort": [{"_score": "desc"}]
                }
            )
            
            # 转换结果格式
            results = []
            for hit in response["hits"]["hits"]:
                results.append((
                    hit["_source"]["file_id"],
                    hit["_source"]["chunk_doc"],
                    hit["_source"]["chunk_metadata"],
                    hit["_score"]
                ))
            
            logger.info(f"ES全文检索返回 {len(results)} 条结果，tags: {tags}")
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

    async def get_summary_id_by_chunk_id(self, file_id: str, chunk_id: str) -> str | None:
        """根据块ID获取摘要ID - 与Oracle接口完全一致"""
        if self.es_client is None:
            return None
        
        try:
            query = {
                "bool": {
                    "must": [
                        {"term": {"file_id": file_id}},
                        {"term": {"chunk_metadata.chunk_type": ChunkType.SUMMARY.value}},
                        {"term": {"chunk_metadata.source_embed_id.keyword": chunk_id}}  # 添加 .keyword
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
        """更新块状态 - 包括对应的summary chunk（如果存在）"""
        if self.es_client is None:
            logger.error("ES客户端未初始化")
            return 0
        
        total_updated = 0
        
        try:
            logger.info(f"开始更新chunk状态: {chunk_id}, status: {status}")
            
            # 第一步：先获取原chunk的file_id
            query_original = {
                "term": {"embed_id": chunk_id}
            }
            
            response_original = await self.es_client.search(
                index=self.index_name,
                body={
                    "query": query_original,
                    "size": 1,
                    "_source": ["file_id"]
                }
            )
            
            logger.info(f"查找原chunk结果: {response_original}")
            
            if not response_original["hits"]["hits"]:
                logger.warning(f"ES未找到chunk_id: {chunk_id} 对应的记录")
                return 0
            
            file_id = response_original["hits"]["hits"][0]["_source"]["file_id"]
            logger.info(f"找到chunk {chunk_id} 对应的file_id: {file_id}")
            
            # 第二步：获取对应的summary chunk ID
            summary_embed_id = await self.get_summary_id_by_chunk_id(file_id, chunk_id)
            logger.info(f"找到summary_embed_id: {summary_embed_id}")
            
            # 第三步：构建需要更新的embed_id列表
            embed_ids_to_update = [chunk_id]
            if summary_embed_id:
                embed_ids_to_update.append(summary_embed_id)
                logger.info(f"将同时更新原chunk {chunk_id} 和summary chunk {summary_embed_id}")
            else:
                logger.info(f"未找到对应的summary chunk，仅更新原chunk {chunk_id}")
            
            # 第四步：批量更新
            query_batch_update = {
                "terms": {"embed_id": embed_ids_to_update}
            }
            
            update_body = {
                "script": {
                    "source": "ctx._source.status = params.status",
                    "params": {"status": status}
                }
            }
            
            logger.info(f"执行批量更新，embed_ids: {embed_ids_to_update}")
            
            response = await self.es_client.update_by_query(
                index=self.index_name,
                body={
                    "query": query_batch_update,
                    "script": update_body["script"]
                },
                refresh=True
            )
            
            total_updated = response["updated"]
            logger.info(f"ES成功更新 {total_updated} 条记录的状态")
            return total_updated
            
        except Exception as e:
            logger.error(f"ES更新状态失败: {e}")
            return 0

    async def get_chunks_by_file_id(self, file_id: str) -> list[KbotBizTxtEmbedding] | None:
        """根据文件ID获取所有chunk - ES版本"""
        if self.es_client is None:
            logger.error("ES客户端未初始化")
            return None
        
        try:
            # 构建查询
            query = {
                "bool": {
                    "must": [
                        {"term": {"file_id": file_id}}
                    ]
                }
            }
            
            # 执行搜索，不限制数量
            response = await self.es_client.search(
                index=self.index_name,
                body={
                    "query": query,
                    "size": 10000,  # 设置一个较大的值，或者可以分页查询
                    "_source": [
                        "embed_id", "kb_id", "file_id", "chunk_doc", 
                        "chunk_metadata", "biz_metadata", "embedding", 
                        "security_level", "status"
                    ]
                }
            )
            
            hits = response["hits"]["hits"]
            if not hits:
                logger.info(f"未找到文件ID为 {file_id} 的chunk")
                return None
            
            chunks = []
            for hit in hits:
                source = hit["_source"]
                
                # 处理embedding字段，确保是列表格式
                # embedding = source.get("embedding")
                # if isinstance(embedding, str):
                #     try:
                #         embedding = json.loads(embedding)
                #     except json.JSONDecodeError:
                #         logger.warning(f"embedding字段解析失败: {embedding}")
                #         embedding = []
                
                # 创建实体对象
                chunk = KbotBizTxtEmbedding(
                    embed_id=source.get("embed_id"),
                    kb_id=source.get("kb_id"),
                    file_id=source.get("file_id", file_id),  # 确保file_id正确
                    chunk_doc=source.get("chunk_doc", ""),
                    chunk_metadata=source.get("chunk_metadata", {}),
                    biz_metadata=source.get("biz_metadata", {}),
                    embedding=[], # embedding 不返回，防止接口数据过大
                    security_level=source.get("security_level", 1),
                    status=source.get("status", 1)
                )
                chunks.append(chunk)
            
            logger.info(f"找到文件ID {file_id} 的 {len(chunks)} 个chunk")
            return chunks
            
        except Exception as e:
            logger.error(f"根据文件ID获取chunk失败: {e}")
            return None
        
    async def get_chunk_doc_by_id(self, embed_id: str) -> str | None:
        """根据ID获取chunk文档 - ES版本"""
        if self.es_client is None:
            logger.error("ES客户端未初始化")
            return None
        
        try:
            # 构建查询
            query = {
                "term": {"embed_id": embed_id}
            }
            
            response = await self.es_client.search(
                index=self.index_name,
                body={
                    "query": query,
                    "size": 1,
                    "_source": ["chunk_doc"]
                }
            )
            
            if response["hits"]["hits"]:
                chunk_doc = response["hits"]["hits"][0]["_source"]["chunk_doc"]
                logger.info(f"ES成功获取chunk文档，ID: {embed_id}")
                return chunk_doc
            
            logger.info(f"ES未找到embed_id: {embed_id} 对应的记录")
            return None
            
        except Exception as e:
            logger.error(f"ES获取chunk文档失败: {e}")
            return None

    async def update_chunk_description(self, embed_id: str, description: str, embeddings: list[float]) -> bool:
        """更新块描述和嵌入向量 - ES版本"""
        if self.es_client is None:
            logger.error("ES客户端未初始化")
            return False
        
        try:
            # 构建更新文档
            update_body = {
                "doc": {
                    "biz_metadata": {
                        "description": description
                    },
                    "embedding": embeddings
                }
            }
            
            response = await self.es_client.update(
                index=self.index_name,
                id=embed_id,
                body=update_body,
                refresh=True
            )
            
            if response["result"] in ["updated", "noop"]:
                logger.info(f"ES成功更新chunk描述和向量，ID: {embed_id}")
                return True
            else:
                logger.error(f"ES更新chunk描述和向量失败: {response['result']}")
                return False
                
        except Exception as e:
            logger.error(f"ES更新chunk描述和向量失败: {e}")
            return False
        
    async def get_all_embeddings(self, 
                           kb_id: int | None = None,
                           page_size: int = 1000,
                           scroll_time: str = "2m") -> list[KbotBizTxtEmbedding]:
        """查询全部记录 - 支持分页滚动获取所有数据
        
        Args:
            kb_id: 知识库ID，如果为None则查询所有kb_id的记录
            page_size: 每页大小，默认1000
            scroll_time: 滚动查询保持时间，默认2分钟
        
        Returns:
            嵌入记录列表
        """
        if self.es_client is None:
            logger.error("ES客户端未初始化")
            return []
        
        try:
            all_embeddings = []
            scroll_id = None
            
            # 构建查询条件
            query = {}
            if kb_id is not None:
                query = {
                    "bool": {
                        "filter": [
                            {"term": {"kb_id": kb_id}}
                        ]
                    }
                }
            else:
                # 如果kb_id为None，查询所有记录
                query = {"match_all": {}}
            
            # 第一次滚动查询
            response = await self.es_client.search(
                index=self.index_name,
                body={
                    "query": query,
                    "size": page_size,
                    "_source": [
                        "embed_id", "kb_id", "file_id", "chunk_doc", 
                        "chunk_metadata", "biz_metadata", "embedding", 
                        "security_level", "status"
                    ]
                },
                scroll=scroll_time
            )
            
            scroll_id = response["_scroll_id"]
            hits = response["hits"]["hits"]
            
            # 处理第一批结果
            while hits:
                for hit in hits:
                    source = hit["_source"]
                    
                    # 创建实体对象
                    embedding = KbotBizTxtEmbedding(
                        embed_id=source.get("embed_id"),
                        kb_id=source.get("kb_id"),
                        file_id=source.get("file_id"),
                        chunk_doc=source.get("chunk_doc", ""),
                        chunk_metadata=source.get("chunk_metadata", {}),
                        biz_metadata=source.get("biz_metadata", {}),
                        embedding=source.get("embedding", []),  # 这里返回完整的嵌入向量
                        security_level=source.get("security_level", 1),
                        status=source.get("status", 1)
                    )
                    all_embeddings.append(embedding)
                
                # 获取下一批结果
                response = await self.es_client.scroll(
                    scroll_id=scroll_id,
                    scroll=scroll_time
                )
                
                scroll_id = response["_scroll_id"]
                hits = response["hits"]["hits"]
            
            # 清理scroll上下文
            if scroll_id:
                await self.es_client.clear_scroll(scroll_id=scroll_id)
            
            logger.info(f"ES成功查询到 {len(all_embeddings)} 条记录，kb_id: {kb_id}")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"ES查询全部记录失败: {e}")
            # 确保清理scroll上下文
            if 'scroll_id' in locals() and scroll_id:
                try:
                    await self.es_client.clear_scroll(scroll_id=scroll_id)
                except Exception:
                    pass
            return []
        
    async def update_tags(self, file_id: str, tags: list[str]) -> bool:
        """根据文件ID更新块标签 - ES版本"""
        if self.es_client is None:
            logger.error("ES客户端未初始化")
            return False
        
        try:
            # 使用update_by_query和script来只更新tags字段，保留其他字段
            response = await self.es_client.update_by_query(
                index=self.index_name,
                body={
                    "query": {
                        "term": {
                            "file_id": file_id
                        }
                    },
                    "script": {
                        "source": """
                            if (ctx._source.biz_metadata == null) {
                                ctx._source.biz_metadata = [:];
                            }
                            if (ctx._source.biz_metadata instanceof Map) {
                                ctx._source.biz_metadata.tags = params.tags;
                            } else {
                                ctx._source.biz_metadata = ['tags': params.tags];
                            }
                        """,
                        "params": {
                            "tags": tags
                        },
                        "lang": "painless"
                    }
                },
                refresh=True
            )
            
            updated_count = response.get("updated", 0)
            return updated_count > 0
                
        except Exception as e:
            logger.error(f"ES更新标签失败: {e}")
            return False