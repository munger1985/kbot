from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import NotFoundError
from loguru import logger
from core.database.vec_elasticsearch import es_client_manager
from dao.entities.kbot_biz_chat_session import KbotBizChatSession, QAData, Reference
from ..common import *
from ..kbot_biz_chat_session_interface import IChatSessionRepository


class ElasticsearchChatSessionRepository(IChatSessionRepository):
    def __init__(self, kb_id: int):
        """
        初始化ES会话仓库
        """
        self.kb_id = kb_id
        self.es_client: AsyncElasticsearch | None = None
        self.session_index = "chat_sessions"
        self.qa_index = "chat_qa_data"

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
                if not await self.es_client.indices.exists(index=self.session_index):
                    await self._create_session_index()
                if not await self.es_client.indices.exists(index=self.qa_index):
                    await self._create_qa_index()
                
                logger.info(f"ES存储库初始化成功，索引: {self.session_index}, {self.qa_index}")
                return True
            else:
                logger.error("ES连接参数为空")
                return False
            
        except Exception as e:
            logger.exception(f"初始化ES连接失败: {e}")
            return False


    async def _create_session_index(self):
        """创建会话索引"""
        mapping = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            },
            "mappings": {
                "properties": {
                    "session_id": {"type": "keyword"},
                    "agent_id": {"type": "integer"},
                    "created_time": {"type": "date"}
                }
            }
        }
        
        try:
            await self.es_client.indices.create(index=self.session_index, body=mapping) # type: ignore
            logger.info(f"创建会话索引成功: {self.session_index}")
        except Exception as e:
            logger.error(f"创建会话索引失败: {e}")
            raise

    async def _create_qa_index(self):
        """创建QA数据索引"""
        mapping = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            },
            "mappings": {
                "properties": {
                    "session_id": {"type": "keyword"},
                    "qa_index": {"type": "integer"},
                    "question": {"type": "text"},
                    "answer": {"type": "text"},
                    "qa_embedding": {"type": "text"},
                    "feedback": {"type": "integer"},
                    "by": {"type": "keyword"},
                    "request_time": {"type": "date"},
                    "response_time": {"type": "date"},
                    "references": {
                        "type": "nested",
                        "properties": {
                            "content": {"type": "text"},
                            "source": {"type": "keyword"}
                        }
                    }
                }
            }
        }
        
        try:
            await self.es_client.indices.create(index=self.qa_index, body=mapping) # type: ignore
            logger.info(f"创建QA索引成功: {self.qa_index}")
        except Exception as e:
            logger.error(f"创建QA索引失败: {e}")
            raise

    async def create_session(self, session_data: KbotBizChatSession) -> bool:
        """
        创建新会话
        :param session_data: 会话数据
        :return: 是否创建成功
        """
        if not self.es_client:
            logger.error("ES客户端未初始化")
            return False

        session_id = session_data.session_id
        
        try:
            # 转换日期格式
            request_time = session_data.qa_data[0].request_time
            if request_time:
                request_time = await es_in_date_format(request_time)

            # 存储会话基本信息
            session_doc = {
                "session_id": session_id,
                "agent_id": session_data.agent_id
            }
            
            await self.es_client.index(
                index=self.session_index,
                id=session_id,
                document=session_doc
            )
            
            # 存储第一个QA对
            qa_data = session_data.qa_data[0]
            
            # 将Reference对象转换为字典
            references_dict = []
            for ref in qa_data.references:
                references_dict.append({
                    "chunk_type": ref.chunk_type,
                    "chunk_file_path": ref.chunk_file_path,
                    "page_num": ref.page_num,
                    "content": ref.content,
                    "download_link": ref.download_link,
                    "preview_link": ref.preview_link,
                    "similarity_score": ref.similarity_score,
                    "reranker_score": ref.reranker_score
                })
            
            qa_doc = {
                "session_id": session_id,
                "qa_index": 0,
                "question": qa_data.question,
                "answer": qa_data.answer,
                "qa_embedding": qa_data.qa_embedding,
                "feedback": qa_data.feedback,
                "by": qa_data.by,
                "request_time": await es_in_date_format(qa_data.request_time) if qa_data.request_time else None, 
                "response_time": await es_in_date_format(qa_data.response_time) if qa_data.response_time else None,
                "references": references_dict  # 使用转换后的字典列表
            }
            
            qa_id = f"{session_id}_0"
            await self.es_client.index(
                index=self.qa_index,
                id=qa_id,
                document=qa_doc
            )
            
            # 刷新索引确保数据可搜索
            await self.es_client.indices.refresh(index=[self.session_index, self.qa_index])
            logger.info(f"创建会话成功: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"创建会话失败: {e}")
            return False

    async def get_session(self, session_id: str) -> KbotBizChatSession | None:
        """
        获取完整会话数据
        :param session_id: 会话ID
        :return: 会话数据或None
        """
        if not self.es_client:
            logger.error("ES客户端未初始化")
            return None
        
        try:
            # 获取会话基本信息
            try:
                session_doc = await self.es_client.get(
                    index=self.session_index,
                    id=session_id
                )
            except NotFoundError:
                logger.debug(f"会话不存在: {session_id}")
                return None
            
            # 获取该会话的所有QA对
            search_body = {
                "query": {
                    "term": {
                        "session_id": session_id
                    }
                },
                "sort": [
                    {"qa_index": {"order": "asc"}}
                ]
            }
            
            qa_response = await self.es_client.search(
                index=self.qa_index,
                body=search_body
            )
            
            qa_data = []
            for hit in qa_response["hits"]["hits"]:
                source = hit["_source"]
                # 将references字典列表转换为Reference对象列表
                reference_objects = []
                for ref_dict in source["references"]:
                    reference_objects.append(Reference(
                        chunk_type=ref_dict["chunk_type"],
                        chunk_file_path=ref_dict["chunk_file_path"],
                        page_num=ref_dict["page_num"],
                        content=ref_dict["content"],
                        download_link=ref_dict["download_link"],
                        preview_link=ref_dict["preview_link"],
                        similarity_score=ref_dict["similarity_score"],
                        reranker_score=ref_dict.get("reranker_score")
                    ))
                
                # 将字典转换为QAData对象
                qa_data.append(QAData(
                    question=source["question"],
                    answer=source["answer"],
                    qa_embedding=source["qa_embedding"],
                    references=reference_objects,
                    feedback=source["feedback"],
                    by=source["by"],
                    request_time=await es_out_date_format(source["request_time"]),
                    response_time=await es_out_date_format(source["response_time"])
                ))
            
            session_source = session_doc["_source"]
            return KbotBizChatSession(
                session_id=session_id,
                agent_id=session_source["agent_id"],
                qa_data=qa_data
            )
            
        except Exception as e:
            logger.error(f"获取会话失败: {e}")
            return None

    async def add_qa_data(self, session_id: str, qa_data: QAData) -> bool:
        """
        向会话添加新的QA对
        :param session_id: 会话ID
        :param qa_data: QA对数据
        :return: 是否添加成功
        """
        if not self.es_client:
            logger.error("ES客户端未初始化")
            return False
        
        try:
            # 检查会话是否存在
            try:
                await self.es_client.get(index=self.session_index, id=session_id)
            except NotFoundError:
                logger.warning(f"会话不存在，无法添加QA数据: {session_id}")
                return False
            
            # 获取当前QA对数量作为新索引
            count_response = await self.es_client.count(
                index=self.qa_index,
                body={
                    "query": {
                        "term": {
                            "session_id": session_id
                        }
                    }
                }
            )
            
            qa_index_num = count_response["count"]
            
            # 存储新的QA对
            qa_doc = {
                "session_id": session_id,
                "qa_index": qa_index_num,
                "question": qa_data.question,
                "answer": qa_data.answer,
                "qa_embedding": qa_data.qa_embedding,
                "feedback": qa_data.feedback,
                "by": qa_data.by,
                "request_time": await es_in_date_format(qa_data.request_time) if qa_data.request_time else None,
                "response_time": await es_in_date_format(qa_data.response_time) if qa_data.response_time else None,
                "references": [ref.to_dict() for ref in qa_data.references]
            }
            
            qa_id = f"{session_id}_{qa_index_num}"
            await self.es_client.index(
                index=self.qa_index,
                id=qa_id,
                document=qa_doc
            )
            
            await self.es_client.indices.refresh(index=self.qa_index)
            logger.debug(f"添加QA数据成功: {session_id}, 索引: {qa_index_num}")
            return True
            
        except Exception as e:
            logger.error(f"添加QA数据失败: {e}")
            return False

    async def delete_session(self, session_id: str) -> bool:
        """
        删除整个会话
        :param session_id: 会话ID
        :return: 是否删除成功
        """
        if not self.es_client:
            logger.error("ES客户端未初始化")
            return False
        
        try:
            # 删除会话记录
            try:
                await self.es_client.delete(index=self.session_index, id=session_id)
            except NotFoundError:
                logger.debug(f"会话记录不存在: {session_id}")
            
            # 删除所有相关的QA记录
            delete_response = await self.es_client.delete_by_query(
                index=self.qa_index,
                body={
                    "query": {
                        "term": {
                            "session_id": session_id
                        }
                    }
                }
            )
            
            await self.es_client.indices.refresh(index=[self.session_index, self.qa_index])
            logger.info(f"删除会话成功: {session_id}, 删除QA记录数: {delete_response.get('deleted', 0)}")
            return True
            
        except Exception as e:
            logger.error(f"删除会话失败: {e}")
            return False

    async def update_qa_feedback(self, session_id: str, qa_index_num: int, feedback: int) -> bool:
        """
        更新QA对的反馈标记
        :param session_id: 会话ID
        :param qa_index_num: QA对索引
        :param feedback: 反馈值
        :return: 是否更新成功
        """
        if not self.es_client:
            logger.error("ES客户端未初始化")
            return False
        
        qa_id = f"{session_id}_{qa_index_num}"
        
        try:
            await self.es_client.update(
                index=self.qa_index,
                id=qa_id,
                body={
                    "doc": {
                        "feedback": feedback
                    }
                }
            )
            logger.debug(f"更新QA反馈成功: {qa_id}, 反馈: {feedback}")
            return True
        except NotFoundError:
            logger.warning(f"QA记录不存在，无法更新反馈: {qa_id}")
            return False
        except Exception as e:
            logger.error(f"更新QA反馈失败: {e}")
            return False

    async def get_last_qa_data(self, session_id: str) -> dict | None:
        """
        获取会话中的最后一个QA对
        :param session_id: 会话ID
        :return: 最后一个QA对数据或None，包含agent_id
        """
        if not self.es_client:
            logger.error("ES客户端未初始化")
            return None
        
        try:
            # 获取会话信息
            try:
                session_doc = await self.es_client.get(index=self.session_index, id=session_id)
            except NotFoundError:
                return None
            
            # 获取最后一个QA对
            search_body = {
                "query": {
                    "term": {
                        "session_id": session_id
                    }
                },
                "sort": [
                    {"qa_index": {"order": "desc"}}
                ],
                "size": 1
            }
            
            qa_response = await self.es_client.search(index=self.qa_index, body=search_body)
            
            if not qa_response["hits"]["hits"]:
                return None
            
            qa_hit = qa_response["hits"]["hits"][0]
            qa_source = qa_hit["_source"]
            session_source = session_doc["_source"]
            
            return {
                "question": qa_source["question"],
                "answer": qa_source["answer"],
                "qa_embedding": qa_source["qa_embedding"],
                "references": qa_source["references"],
                "feedback": qa_source["feedback"],
                "by": qa_source["by"],
                "request_time": await es_out_date_format(qa_source["request_time"]),
                "response_time": await es_out_date_format(qa_source["response_time"]),
                "agent_id": session_source["agent_id"]
            }
            
        except Exception as e:
            logger.error(f"获取最后一个QA数据失败: {e}")
            return None

    async def get_qa_data(self, session_id: str, qa_index_num: int) -> QAData | None:
        """
        获取单个QA对
        :param session_id: 会话ID
        :param qa_index_num: QA对索引
        :return: QA对数据或None
        """
        if not self.es_client:
            logger.error("ES客户端未初始化")
            return None
        
        qa_id = f"{session_id}_{qa_index_num}"
        
        try:
            qa_doc = await self.es_client.get(index=self.qa_index, id=qa_id)
            qa_source = qa_doc["_source"]
            
            # 将references字典列表转换为Reference对象列表
            reference_objects = []
            for ref_dict in qa_source["references"]:
                reference_objects.append(Reference(
                    chunk_type=ref_dict["chunk_type"],
                    chunk_file_path=ref_dict["chunk_file_path"],
                    page_num=ref_dict["page_num"],
                    content=ref_dict["content"],
                    download_link=ref_dict["download_link"],
                    preview_link=ref_dict["preview_link"],
                    similarity_score=ref_dict["similarity_score"],
                    reranker_score=ref_dict.get("reranker_score")
                ))
            
            return QAData(
                question=qa_source["question"],
                answer=qa_source["answer"],
                qa_embedding=qa_source["qa_embedding"],
                references=reference_objects,
                feedback=qa_source["feedback"],
                by=qa_source["by"],
                request_time=await es_out_date_format(qa_source["request_time"]),
                response_time=await es_out_date_format(qa_source["response_time"])
            )
        except NotFoundError:
            return None
        except Exception as e:
            logger.error(f"获取QA数据失败: {e}")
            return None

    async def update_last_qa_data_answer(self, session_id: str, answer: str) -> bool:
        """
        更新会话中的最后一个QA对的答案
        :param session_id: 会话ID
        :param answer: 答案
        :return: 是否更新成功
        """
        if not self.es_client:
            logger.error("ES客户端未初始化")
            return False
        
        try:
            # 获取最后一个QA对的ID
            search_body = {
                "query": {
                    "term": {
                        "session_id": session_id
                    }
                },
                "sort": [
                    {"qa_index": {"order": "desc"}}
                ],
                "size": 1
            }
            
            qa_response = await self.es_client.search(index=self.qa_index, body=search_body)
            
            if not qa_response["hits"]["hits"]:
                logger.warning(f"未找到QA数据，无法更新答案: {session_id}")
                return False
            
            qa_hit = qa_response["hits"]["hits"][0]
            qa_id = qa_hit["_id"]
            
            # 更新答案
            await self.es_client.update(
                index=self.qa_index,
                id=qa_id,
                body={
                    "doc": {
                        "answer": answer
                    }
                }
            )
            await self.es_client.indices.refresh(index=[self.qa_index])
            logger.debug(f"更新最后一个QA答案成功: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"更新最后一个QA答案失败: {e}")
            return False

    async def delete_by_agent_id(self, agent_id: int) -> bool:
        """
        根据agent_id删除所有相关的会话和QA数据
        :param agent_id: 要删除的agent ID
        :return: 是否删除成功
        """
        if not self.es_client:
            logger.error("ES客户端未初始化")
            return False
        
        try:
            # 查找所有匹配的会话
            search_body = {
                "query": {
                    "term": {
                        "agent_id": agent_id
                    }
                }
            }
            
            session_response = await self.es_client.search(
                index=self.session_index,
                body=search_body,
                size=1000
            )
            
            if not session_response["hits"]["hits"]:
                logger.info(f"未找到agent_id相关的会话: {agent_id}")
                return True
            
            # 收集所有会话ID
            session_ids = [hit["_source"]["session_id"] for hit in session_response["hits"]["hits"]]
            
            # 删除所有会话记录
            session_delete_response = await self.es_client.delete_by_query(
                index=self.session_index,
                body={
                    "query": {
                        "term": {
                            "agent_id": agent_id
                        }
                    }
                }
            )
            
            # 删除所有相关的QA记录
            qa_delete_count = 0
            for session_id in session_ids:
                qa_response = await self.es_client.delete_by_query(
                    index=self.qa_index,
                    body={
                        "query": {
                            "term": {
                                "session_id": session_id
                            }
                        }
                    }
                )
                qa_delete_count += qa_response.get('deleted', 0)
            
            await self.es_client.indices.refresh(index=[self.session_index, self.qa_index])
            
            logger.info(f"根据agent_id删除数据成功: {agent_id}, 删除会话数: {session_delete_response.get('deleted', 0)}, 删除QA记录数: {qa_delete_count}")
            return True
            
        except Exception as e:
            logger.error(f"根据agent_id删除数据失败: {e}")
            return False