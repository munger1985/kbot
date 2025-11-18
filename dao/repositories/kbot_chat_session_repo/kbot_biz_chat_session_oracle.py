import oracledb
from loguru import logger
from core.database.vec_oracle_pool import OracleConnParams, AsyncOracleConnectionPoolManager
from dao.entities.kbot_biz_chat_session import KbotBizChatSession, QAData, Reference
from utils.oracle_vec_handler import OracleVecHandler
from utils.common import safe_read_content
from ..common import *
from ..kbot_biz_chat_session_interface import IChatSessionRepository


class OracleChatSessionRepository(IChatSessionRepository):
    """Oracle 23ai 会话存储库"""
    def __init__(self, kb_id: int):
        self.kb_id = kb_id
        self.db_conf = None
        self.conn_params = None
        self.pool_manager = AsyncOracleConnectionPoolManager()

    async def initialize(self, connstr: dict) -> bool:

        if connstr is not None:
            username = connstr.get("user")
            password = connstr.get("password")
            if username is None or password is None:
                logger.error("Oracle连接参数中缺少用户名或密码")
                return False
            
            self.conn_params = OracleConnParams(
                user=username,
                password=password,
                dsn=f"{connstr.get('host')}:{connstr.get('port')}/{connstr.get('service_name')}:pooled"
            )
            return True
        else:
            logger.error("Oracle连接参数为空")
            return False

    async def create_session(self, session_data: KbotBizChatSession) -> bool:
        """
        创建新会话
        :param session_data: 会话数据
        :return: 是否创建成功
        """
        if self.conn_params is None:
            logger.error("Oracle连接参数未初始化")
            return False

        try:
            async with self.pool_manager.get_connection_ctx(self.conn_params) as conn:
                cursor = conn.cursor()
                
                if self.pool_manager._loop is None:
                    logger.error("连接池事件循环不存在")
                    return False
                
                # 插入第一个QA数据
                qa_data = session_data.qa_data[0]
                sql_qa = """
                    INSERT INTO kbot_biz_chat_session (
                        session_id, agent_id, question, answer, qa_embedding, 
                        feedback, username, request_time, response_time
                    ) VALUES (
                        :session_id, :agent_id, :question, :answer, :qa_embedding,
                        :feedback, :username, TO_TIMESTAMP(:request_time, 'YYYY-MM-DD HH24:MI:SS.FF'), 
                        TO_TIMESTAMP(:response_time, 'YYYY-MM-DD HH24:MI:SS.FF')
                    ) RETURNING qa_id INTO :qa_id
                """
                
                # 准备QA参数
                qa_params = {
                    "session_id": session_data.session_id,
                    "agent_id": session_data.agent_id,
                    "question": qa_data.question,
                    "answer": qa_data.answer,
                    "qa_embedding": OracleVecHandler().convert(vec=qa_data.qa_embedding, to_string=True),
                    "feedback": qa_data.feedback,
                    "username": qa_data.by,
                    "request_time": qa_data.request_time,
                    "response_time": qa_data.response_time,
                    "qa_id": cursor.var(oracledb.NUMBER)
                }
                
                # 执行QA插入
                await self.pool_manager._loop.run_in_executor(None, cursor.execute, sql_qa, qa_params)
                qa_id = qa_params["qa_id"].getvalue()[0]
                
                # 插入参考文献数据
                if qa_data.references:
                    sql_ref = """
                        INSERT INTO kbot_biz_chat_references (
                            qa_id, chunk_type, chunk_file_path, page_num,
                            chunk_content, download_link, preview_link, similarity_score, reranker_score
                        ) VALUES (
                            :qa_id, :chunk_type, :chunk_file_path, :page_num,
                            :chunk_content, :download_link, :preview_link, :similarity_score, :reranker_score
                        )
                    """
                    
                    ref_data = []
                    for ref in qa_data.references:
                        ref_data.append((
                            qa_id,
                            ref.chunk_type,
                            ref.chunk_file_path,
                            ref.page_num,
                            ref.content,
                            ref.download_link,
                            ref.preview_link,
                            ref.similarity_score,
                            ref.reranker_score
                        ))
                    
                    # 批量插入参考文献
                    await self.pool_manager._loop.run_in_executor(
                        None, cursor.executemany, sql_ref, ref_data
                    )
                
                # 提交事务
                await self.pool_manager._loop.run_in_executor(None, conn.commit)
                logger.info(f"创建会话成功: {session_data.session_id}, QA_ID: {qa_id}")
                return True
                
        except oracledb.Error as e:
            logger.error(f"创建会话失败: {e}")
            return False
        except Exception as e:
            logger.error(f"创建会话过程中发生未知错误: {e}")
            return False

    async def get_session(self, session_id: str) -> KbotBizChatSession | None:
        """
        获取完整会话数据
        :param session_id: 会话ID
        :return: 会话数据或None
        """
        if self.conn_params is None:
            logger.error("Oracle连接参数未初始化")
            return None
        
        try:
            sql = """
                SELECT 
                    s.agent_id,
                    s.question, s.answer, s.qa_embedding, s.feedback, s.username,
                    s.request_time, s.response_time,
                    r.chunk_type, r.chunk_file_path, r.page_num,
                    r.chunk_content, r.download_link, r.preview_link,
                    r.similarity_score, r.reranker_score
                FROM kbot_biz_chat_session s
                LEFT JOIN kbot_biz_chat_references r ON s.qa_id = r.qa_id
                WHERE s.session_id = :session_id
                ORDER BY s.request_time
            """
            
            params = {"session_id": session_id}
            result = await self.pool_manager.query(self.conn_params, sql, params)
            
            if not result:
                logger.debug(f"会话不存在: {session_id}")
                return None
            
            # 按QA分组处理数据
            qa_data_map = {}
            for row in result:
                # 提取QA基本信息
                agent_id = row[0]
                question = safe_read_content(row[1])  # CLOB字段需要安全读取
                answer = safe_read_content(row[2])    # CLOB字段需要安全读取
                qa_embedding_str = row[3]
                feedback = row[4]
                username = row[5]
                request_time = row[6]
                response_time = row[7]
                
                qa_key = f"{question}_{request_time}"
                if qa_key not in qa_data_map:
                    # 转换向量数据
                    qa_embedding = []
                    if qa_embedding_str:
                        qa_embedding = OracleVecHandler().convert(vec=qa_embedding_str, to_string=False)
                    
                    qa_data_map[qa_key] = {
                        "question": question,
                        "answer": answer,
                        "qa_embedding": qa_embedding,
                        "references": [],
                        "feedback": feedback,
                        "by": username,
                        "request_time": request_time.strftime("%Y-%m-%d %H:%M:%S.%f") if request_time else None,
                        "response_time": response_time.strftime("%Y-%m-%d %H:%M:%S.%f") if response_time else None
                    }
                
                # 添加参考文献数据（如果有）
                if row[8] is not None:  # chunk_type不为空表示有参考文献
                    ref = Reference(
                        chunk_type=row[8],
                        chunk_file_path=row[9],
                        page_num=row[10],
                        content=safe_read_content(row[11]),
                        download_link=row[12],
                        preview_link=row[13],
                        similarity_score=row[14],
                        reranker_score=row[15]
                    )
                    qa_data_map[qa_key]["references"].append(ref)
            
            # 构建QAData列表
            qa_data_list = []
            for qa_info in qa_data_map.values():
                qa_data_list.append(QAData(
                    question=qa_info["question"],
                    answer=qa_info["answer"],
                    qa_embedding=qa_info["qa_embedding"],
                    references=qa_info["references"],
                    feedback=qa_info["feedback"],
                    by=qa_info["by"],
                    request_time=qa_info["request_time"],
                    response_time=qa_info["response_time"]
                ))
            
            return KbotBizChatSession(
                session_id=session_id,
                agent_id=agent_id,
                qa_data=qa_data_list
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
        if self.conn_params is None:
            logger.error("Oracle连接参数未初始化")
            return False
        
        try:
            async with self.pool_manager.get_connection_ctx(self.conn_params) as conn:
                cursor = conn.cursor()
                
                if self.pool_manager._loop is None:
                    logger.error("连接池事件循环不存在")
                    return False
                
                # 插入QA数据
                sql_qa = """
                    INSERT INTO kbot_biz_chat_session (
                        session_id, agent_id, question, answer, qa_embedding, 
                        feedback, username, request_time, response_time
                    ) VALUES (
                        :session_id, 
                        (SELECT agent_id FROM kbot_biz_chat_session WHERE session_id = :session_id FETCH FIRST 1 ROWS ONLY),
                        :question, :answer, :qa_embedding,
                        :feedback, :username, TO_TIMESTAMP(:request_time, 'YYYY-MM-DD HH24:MI:SS.FF'), 
                        TO_TIMESTAMP(:response_time, 'YYYY-MM-DD HH24:MI:SS.FF')
                    ) RETURNING qa_id INTO :qa_id
                """
                
                qa_params = {
                    "session_id": session_id,
                    "question": qa_data.question,
                    "answer": qa_data.answer,
                    "qa_embedding": OracleVecHandler().convert(vec=qa_data.qa_embedding, to_string=True),
                    "feedback": qa_data.feedback,
                    "username": qa_data.by,
                    "request_time": qa_data.request_time,
                    "response_time": qa_data.response_time,
                    "qa_id": cursor.var(oracledb.NUMBER)
                }
                
                await self.pool_manager._loop.run_in_executor(None, cursor.execute, sql_qa, qa_params)
                qa_id = qa_params["qa_id"].getvalue()[0]
                
                # 插入参考文献数据
                if qa_data.references:
                    sql_ref = """
                        INSERT INTO kbot_biz_chat_references (
                            qa_id, chunk_type, chunk_file_path, page_num,
                            chunk_content, download_link, preview_link, similarity_score, reranker_score
                        ) VALUES (
                            :qa_id, :chunk_type, :chunk_file_path, :page_num,
                            :chunk_content, :download_link, :preview_link, :similarity_score, :reranker_score
                        )
                    """
                    
                    ref_data = []
                    for ref in qa_data.references:
                        ref_data.append((
                            qa_id,
                            ref.chunk_type,
                            ref.chunk_file_path,
                            ref.page_num,
                            ref.content,
                            ref.download_link,
                            ref.preview_link,
                            ref.similarity_score,
                            ref.reranker_score
                        ))
                    
                    await self.pool_manager._loop.run_in_executor(
                        None, cursor.executemany, sql_ref, ref_data
                    )
                
                await self.pool_manager._loop.run_in_executor(None, conn.commit)
                logger.debug(f"添加QA数据成功: {session_id}, QA_ID: {qa_id}")
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
        if self.conn_params is None:
            logger.error("Oracle连接参数未初始化")
            return False
        
        try:
            async with self.pool_manager.get_connection_ctx(self.conn_params) as conn:
                cursor = conn.cursor()
                
                if self.pool_manager._loop is None:
                    logger.error("连接池事件循环不存在")
                    return False
                
                # 先删除参考文献数据
                sql_delete_refs = """
                    DELETE FROM kbot_biz_chat_references 
                    WHERE qa_id IN (SELECT qa_id FROM kbot_biz_chat_session WHERE session_id = :session_id)
                """
                await self.pool_manager._loop.run_in_executor(
                    None, cursor.execute, sql_delete_refs, {"session_id": session_id}
                )
                
                # 删除会话数据
                sql_delete_session = """
                    DELETE FROM kbot_biz_chat_session WHERE session_id = :session_id
                """
                await self.pool_manager._loop.run_in_executor(
                    None, cursor.execute, sql_delete_session, {"session_id": session_id}
                )
                
                await self.pool_manager._loop.run_in_executor(None, conn.commit)
                logger.info(f"删除会话成功: {session_id}")
                return True
                
        except Exception as e:
            logger.error(f"删除会话失败: {e}")
            return False

    async def update_qa_feedback(self, session_id: str, qa_index_num: int, feedback: int) -> bool:
        """
        更新QA对的反馈标记
        :param session_id: 会话ID
        :param qa_index_num: QA对索引（在Oracle中通过request_time或其他标识来定位）
        :param feedback: 反馈值
        :return: 是否更新成功
        """
        if self.conn_params is None:
            logger.error("Oracle连接参数未初始化")
            return False
        
        try:
            # 由于Oracle表结构没有qa_index，我们通过其他方式定位特定的QA记录
            # 这里假设通过session_id和按时间排序后的位置来定位
            sql = """
                UPDATE kbot_biz_chat_session 
                SET feedback = :feedback 
                WHERE qa_id = (
                    SELECT qa_id FROM (
                        SELECT qa_id, ROW_NUMBER() OVER (ORDER BY request_time) as rn
                        FROM kbot_biz_chat_session 
                        WHERE session_id = :session_id
                    ) WHERE rn = :qa_index_num
                )
            """
            
            params = {
                "session_id": session_id,
                "qa_index_num": qa_index_num + 1,  # ROW_NUMBER从1开始
                "feedback": feedback
            }
            
            result = await self.pool_manager.execute_dml(self.conn_params, sql, params)
            if result > 0:
                logger.debug(f"更新QA反馈成功: {session_id}, 索引: {qa_index_num}, 反馈: {feedback}")
                return True
            else:
                logger.warning(f"QA记录不存在，无法更新反馈: {session_id}, 索引: {qa_index_num}")
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
        if self.conn_params is None:
            logger.error("Oracle连接参数未初始化")
            return None
        
        try:
            sql = """
                SELECT 
                    s.agent_id, s.question, s.answer, s.qa_embedding, s.feedback, 
                    s.username, s.request_time, s.response_time
                FROM kbot_biz_chat_session s
                WHERE s.session_id = :session_id
                ORDER BY s.request_time DESC
                FETCH FIRST 1 ROWS ONLY
            """
            
            params = {"session_id": session_id}
            result = await self.pool_manager.query(self.conn_params, sql, params)
            
            if not result:
                return None
            
            row = result[0]
            # 获取参考文献数据
            refs_sql = """
                SELECT chunk_type, chunk_file_path, page_num, chunk_content,
                       download_link, preview_link, similarity_score, reranker_score
                FROM kbot_biz_chat_references 
                WHERE qa_id = (
                    SELECT qa_id FROM kbot_biz_chat_session 
                    WHERE session_id = :session_id 
                    ORDER BY request_time DESC 
                    FETCH FIRST 1 ROWS ONLY
                )
            """
            refs_result = await self.pool_manager.query(self.conn_params, refs_sql, params)
            
            references = []
            for ref_row in refs_result:
                references.append(Reference(
                    chunk_type=ref_row[0],
                    chunk_file_path=ref_row[1],
                    page_num=ref_row[2],
                    content=safe_read_content(ref_row[3]),
                    download_link=ref_row[4],
                    preview_link=ref_row[5],
                    similarity_score=ref_row[6],
                    reranker_score=ref_row[7]
                ))
            
            # 转换向量数据
            qa_embedding = []
            if row[3]:  # qa_embedding字段
                qa_embedding = OracleVecHandler().convert(vec=row[3], to_string=False)
                # 确保qa_embedding是Python列表，而不是array.array
                if hasattr(qa_embedding, 'tolist'):
                    qa_embedding = qa_embedding.tolist() # type: ignore
                elif isinstance(qa_embedding, list):
                    qa_embedding = list(qa_embedding)
            
            # 将Reference对象转换为字典
            references_dict = [ref.to_dict() for ref in references]
            
            return {
                "question": safe_read_content(row[1]),  # CLOB字段需要安全读取
                "answer": safe_read_content(row[2]),    # CLOB字段需要安全读取
                "qa_embedding": qa_embedding,
                "references": references_dict,
                "feedback": row[4],
                "by": row[5],
                "request_time": row[6].strftime("%Y-%m-%d %H:%M:%S.%f") if row[6] else None,
                "response_time": row[7].strftime("%Y-%m-%d %H:%M:%S.%f") if row[7] else None,
                "agent_id": row[0]
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
        if self.conn_params is None:
            logger.error("Oracle连接参数未初始化")
            return None
        
        try:
            sql = """
                SELECT 
                    question, answer, qa_embedding, feedback, username,
                    request_time, response_time
                FROM (
                    SELECT 
                        question, answer, qa_embedding, feedback, username,
                        request_time, response_time,
                        ROW_NUMBER() OVER (ORDER BY request_time) as rn
                    FROM kbot_biz_chat_session 
                    WHERE session_id = :session_id
                ) WHERE rn = :qa_index_num
            """
            
            params = {
                "session_id": session_id,
                "qa_index_num": qa_index_num + 1  # ROW_NUMBER从1开始
            }
            
            result = await self.pool_manager.query(self.conn_params, sql, params)
            
            if not result:
                return None
            
            row = result[0]
            
            # 获取参考文献数据
            refs_sql = """
                SELECT chunk_type, chunk_file_path, page_num, chunk_content,
                       download_link, preview_link, similarity_score, reranker_score
                FROM kbot_biz_chat_references 
                WHERE qa_id = (
                    SELECT qa_id FROM (
                        SELECT qa_id, ROW_NUMBER() OVER (ORDER BY request_time) as rn
                        FROM kbot_biz_chat_session 
                        WHERE session_id = :session_id
                    ) WHERE rn = :qa_index_num
                )
            """
            refs_result = await self.pool_manager.query(self.conn_params, refs_sql, params)
            
            references = []
            for ref_row in refs_result:
                references.append(Reference(
                    chunk_type=ref_row[0],
                    chunk_file_path=ref_row[1],
                    page_num=ref_row[2],
                    content=safe_read_content(ref_row[3]),
                    download_link=ref_row[4],
                    preview_link=ref_row[5],
                    similarity_score=ref_row[6],
                    reranker_score=ref_row[7]
                ))
            
            # 转换向量数据
            qa_embedding = []
            if row[2]:  # qa_embedding字段
                qa_embedding = OracleVecHandler().convert(vec=row[2], to_string=False)
            
            return QAData(
                question=safe_read_content(row[0]),  # CLOB字段需要安全读取
                answer=safe_read_content(row[1]),    # CLOB字段需要安全读取
                qa_embedding=qa_embedding, # type: ignore
                references=references,
                feedback=row[3],
                by=row[4],
                request_time=row[5].strftime("%Y-%m-%d %H:%M:%S.%f") if row[5] else None,
                response_time=row[6].strftime("%Y-%m-%d %H:%M:%S.%f") if row[6] else None
            )
            
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
        if self.conn_params is None:
            logger.error("Oracle连接参数未初始化")
            return False
        
        try:
            sql = """
                UPDATE kbot_biz_chat_session 
                SET answer = :answer 
                WHERE qa_id = (
                    SELECT qa_id FROM kbot_biz_chat_session 
                    WHERE session_id = :session_id 
                    ORDER BY request_time DESC 
                    FETCH FIRST 1 ROWS ONLY
                )
            """
            
            params = {
                "session_id": session_id,
                "answer": answer
            }
            
            result = await self.pool_manager.execute_dml(self.conn_params, sql, params)
            if result > 0:
                logger.debug(f"更新最后一个QA答案成功: {session_id}")
                return True
            else:
                logger.warning(f"未找到QA数据，无法更新答案: {session_id}")
                return False
                
        except Exception as e:
            logger.error(f"更新最后一个QA答案失败: {e}")
            return False

    async def delete_by_agent_id(self, agent_id: int) -> bool:
        """
        根据agent_id删除所有相关的会话和QA数据
        :param agent_id: 要删除的agent ID
        :return: 是否删除成功
        """
        if self.conn_params is None:
            logger.error("Oracle连接参数未初始化")
            return False
        
        try:
            async with self.pool_manager.get_connection_ctx(self.conn_params) as conn:
                cursor = conn.cursor()
                
                if self.pool_manager._loop is None:
                    logger.error("连接池事件循环不存在")
                    return False
                
                # 先删除参考文献数据
                sql_delete_refs = """
                    DELETE FROM kbot_biz_chat_references 
                    WHERE qa_id IN (SELECT qa_id FROM kbot_biz_chat_session WHERE agent_id = :agent_id)
                """
                await self.pool_manager._loop.run_in_executor(
                    None, cursor.execute, sql_delete_refs, {"agent_id": agent_id}
                )
                
                # 删除会话数据
                sql_delete_session = """
                    DELETE FROM kbot_biz_chat_session WHERE agent_id = :agent_id
                """
                await self.pool_manager._loop.run_in_executor(
                    None, cursor.execute, sql_delete_session, {"agent_id": agent_id}
                )
                
                await self.pool_manager._loop.run_in_executor(None, conn.commit)
                logger.info(f"根据agent_id删除数据成功: {agent_id}")
                return True
                
        except Exception as e:
            logger.error(f"根据agent_id删除数据失败: {e}")
            return False