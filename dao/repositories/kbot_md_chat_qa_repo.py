from dao.entities.kbot_md_chat_qa import KbotMdChatQa, KbotMdChatReferences
from sqlalchemy import select, update, delete, func
from utils.oracle_vec_handler import OracleVecHandler
from core.database.meta_oracle import get_session
from utils.common import safe_read_content
from dao.entities.kbot_biz_chat_session import QAData, Reference, KbotBizChatSession
from loguru import logger
from datetime import datetime


class KbotMdChatQaRepository():
    """Oracle 23ai 会话存储库 - SQLAlchemy 2.0 ORM版本"""

    def _parse_datetime(self, datetime_str: str | None) -> datetime | None:
        """将日期时间字符串转换为 datetime 对象"""
        if not datetime_str:
            return None
        
        try:
            # 支持多种日期时间格式
            formats = [
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M",
                "%Y-%m-%d"
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(datetime_str, fmt)
                except ValueError:
                    continue
            
            # 如果都不匹配，尝试自动解析
            return datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
        except Exception as e:
            logger.warning(f"日期时间解析失败: {datetime_str}, 错误: {e}")
            return None
        
    async def create_session(self, session_data: KbotBizChatSession) -> bool:
        """
        创建新会话 - 手动关联版本
        """
        try:
            async with get_session() as session:
                qa_data = session_data.qa_data[0]
                
                # 转换日期时间
                request_time = self._parse_datetime(qa_data.request_time)
                response_time = self._parse_datetime(qa_data.response_time)

                # 创建会话记录 - 只使用ORM模型中定义的字段
                chat_session = KbotMdChatQa(
                    session_id=session_data.session_id,
                    agent_id=session_data.agent_id,
                    question=qa_data.question,
                    answer=qa_data.answer,
                    qa_embedding=OracleVecHandler().convert(vec=qa_data.qa_embedding, to_string=True),
                    feedback=qa_data.feedback,
                    username=qa_data.by,
                    request_time=request_time,
                    response_time=response_time
                )
                
                # 添加会话到数据库
                session.add(chat_session)
                await session.flush()  # 刷新以获取生成的 qa_id
                
                # 添加参考文献
                if qa_data.references:
                    for ref in qa_data.references:
                        reference = KbotMdChatReferences(
                            qa_id=chat_session.qa_id,  # 手动设置关联
                            chunk_type=ref.chunk_type,
                            chunk_file_path=ref.chunk_file_path,
                            page_num=ref.page_num,
                            chunk_content=ref.content,
                            download_link=ref.download_link,
                            preview_link=ref.preview_link,
                            similarity_score=ref.similarity_score,
                            reranker_score=ref.reranker_score
                        )
                        session.add(reference)
                
                await session.commit()
                logger.info(f"创建会话成功: {session_data.session_id}, QA_ID: {chat_session.qa_id}")
                return True
                
        except Exception as e:
            logger.error(f"创建会话失败: {e}")
            return False

    async def add_qa_data(self, session_id: str, qa_data: QAData) -> bool:
        """
        向会话添加新的QA对 - 手动关联版本
        """
        try:
            async with get_session() as session:
                # 获取agent_id
                stmt = select(KbotMdChatQa.agent_id).where(
                    KbotMdChatQa.session_id == session_id
                ).limit(1)
                result = await session.execute(stmt)
                agent_row = result.scalar_one_or_none()
                
                if not agent_row:
                    logger.error(f"未找到会话对应的agent_id: {session_id}")
                    return False
                
                # 转换日期时间
                request_time = self._parse_datetime(qa_data.request_time)
                response_time = self._parse_datetime(qa_data.response_time)
                
                # 创建新的QA记录 - 只使用ORM模型中定义的字段
                chat_session = KbotMdChatQa(
                    session_id=session_id,
                    agent_id=agent_row,
                    question=qa_data.question,
                    answer=qa_data.answer,
                    qa_embedding=OracleVecHandler().convert(vec=qa_data.qa_embedding, to_string=True),
                    feedback=qa_data.feedback,
                    username=qa_data.by,
                    request_time=request_time,
                    response_time=response_time
                )
                
                session.add(chat_session)
                await session.flush()  # 刷新以获取生成的 qa_id
                
                # 添加参考文献
                if qa_data.references:
                    for ref in qa_data.references:
                        reference = KbotMdChatReferences(
                            qa_id=chat_session.qa_id,  # 手动设置关联
                            chunk_type=ref.chunk_type,
                            chunk_file_path=ref.chunk_file_path,
                            page_num=ref.page_num,
                            chunk_content=ref.content,
                            download_link=ref.download_link,
                            preview_link=ref.preview_link,
                            similarity_score=ref.similarity_score,
                            reranker_score=ref.reranker_score
                        )
                        session.add(reference)
                
                await session.commit()
                logger.debug(f"添加QA数据成功: {session_id}, QA_ID: {chat_session.qa_id}")
                return True
                
        except Exception as e:
            logger.error(f"添加QA数据失败: {e}")
            return False

    async def get_session(self, session_id: str) -> dict | None:
        """
        获取完整会话数据 - SQLAlchemy版本
        :param session_id: 会话ID
        :return: 会话数据或None
        """
        try:
            async with get_session() as session:
                # 获取该会话的所有QA记录
                stmt = select(KbotMdChatQa).where(
                    KbotMdChatQa.session_id == session_id
                ).order_by(
                    KbotMdChatQa.request_time
                )

                result = await session.execute(stmt)
                session_rows = result.scalars().all()

                if not session_rows:
                    logger.debug(f"会话不存在: {session_id}")
                    return None

                # 按QA分组处理数据
                qa_data_map = {}
                for session_row in session_rows:
                    # 提取QA基本信息
                    agent_id = session_row.agent_id
                    question = safe_read_content(session_row.question)
                    answer = safe_read_content(session_row.answer)
                    qa_embedding_str = session_row.qa_embedding
                    feedback = session_row.feedback
                    username = session_row.username
                    request_time = session_row.request_time
                    response_time = session_row.response_time
                    
                    qa_key = f"{question}_{request_time}"
                    if qa_key not in qa_data_map:
                        # 转换向量数据
                        qa_embedding = []
                        if qa_embedding_str:
                            qa_embedding = OracleVecHandler().convert(vec=qa_embedding_str, to_string=False)
                            # 确保qa_embedding是Python列表
                            if hasattr(qa_embedding, 'tolist'):
                                qa_embedding = qa_embedding.tolist()  # type: ignore
                            elif isinstance(qa_embedding, list):
                                qa_embedding = list(qa_embedding)
                        
                        qa_data_map[qa_key] = {
                            "question": question,
                            "answer": answer,
                            "qa_embedding": qa_embedding,
                            "references": [],
                            "feedback": feedback,
                            "by": username,
                            "request_time": request_time.strftime("%Y-%m-%d %H:%M:%S.%f") if request_time else None, # type: ignore
                            "response_time": response_time.strftime("%Y-%m-%d %H:%M:%S.%f") if response_time else None # type: ignore
                        }
                    
                    # 获取对应的参考文献
                    ref_stmt = select(KbotMdChatReferences).where(
                        KbotMdChatReferences.qa_id == session_row.qa_id
                    )
                    ref_result = await session.execute(ref_stmt)
                    references_rows = ref_result.scalars().all()
                    
                    # 添加参考文献数据（如果有）
                    for ref_row in references_rows:
                        ref = Reference(
                            chunk_type=ref_row.chunk_type,
                            chunk_file_path=ref_row.chunk_file_path,
                            page_num=ref_row.page_num,
                            content=safe_read_content(ref_row.chunk_content),
                            download_link=ref_row.download_link,
                            preview_link=ref_row.preview_link,
                            similarity_score=ref_row.similarity_score,
                            reranker_score=ref_row.reranker_score
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
                
                return {
                    "session_id": session_id,
                    "agent_id": agent_id,
                    "qa_data": qa_data_list
                }
                
        except Exception as e:
            logger.error(f"获取会话失败: {e}")
            return None

    async def delete_session(self, session_id: str) -> bool:
        """
        删除整个会话 - 手动关联版本
        """
        try:
            async with get_session() as session:
                # 先获取所有相关的qa_id
                stmt = select(KbotMdChatQa.qa_id).where(
                    KbotMdChatQa.session_id == session_id
                )
                result = await session.execute(stmt)
                qa_ids = result.scalars().all()
                
                if qa_ids:
                    # 删除参考文献
                    ref_delete_stmt = delete(KbotMdChatReferences).where(
                        KbotMdChatReferences.qa_id.in_(qa_ids)
                    )
                    await session.execute(ref_delete_stmt)
                    
                    # 删除会话记录
                    session_delete_stmt = delete(KbotMdChatQa).where(
                        KbotMdChatQa.session_id == session_id
                    )
                    await session.execute(session_delete_stmt)
                
                await session.commit()
                logger.info(f"删除会话成功: {session_id}")
                return True
                
        except Exception as e:
            logger.error(f"删除会话失败: {e}")
            return False

    async def delete_by_agent_id(self, agent_id: int) -> bool:
        """
        根据agent_id删除所有相关的会话和QA数据 - 手动关联版本
        """
        try:
            async with get_session() as session:
                # 先获取所有相关的qa_id
                stmt = select(KbotMdChatQa.qa_id).where(
                    KbotMdChatQa.agent_id == agent_id
                )
                result = await session.execute(stmt)
                qa_ids = result.scalars().all()
                
                if qa_ids:
                    # 删除参考文献
                    ref_delete_stmt = delete(KbotMdChatReferences).where(
                        KbotMdChatReferences.qa_id.in_(qa_ids)
                    )
                    await session.execute(ref_delete_stmt)
                    
                    # 删除会话记录
                    session_delete_stmt = delete(KbotMdChatQa).where(
                        KbotMdChatQa.agent_id == agent_id
                    )
                    await session.execute(session_delete_stmt)
                
                await session.commit()
                logger.info(f"根据agent_id删除数据成功: {agent_id}")
                return True
                
        except Exception as e:
            logger.error(f"根据agent_id删除数据失败: {e}")
            return False

    async def update_qa_feedback(self, session_id: str, qa_index_num: int, feedback: int) -> bool:
        """
        更新QA对的反馈标记 - SQLAlchemy版本
        :param session_id: 会话ID
        :param qa_index_num: QA对索引
        :param feedback: 反馈值
        :return: 是否更新成功
        """
        try:
            async with get_session() as session:
                # 使用窗口函数获取指定索引的QA记录
                subquery = select(
                    KbotMdChatQa.qa_id,
                    func.row_number().over(
                        order_by=KbotMdChatQa.request_time
                    ).label('rn')
                ).where(
                    KbotMdChatQa.session_id == session_id
                ).subquery()

                # 更新指定索引的QA反馈
                update_stmt = update(KbotMdChatQa).where(
                    KbotMdChatQa.qa_id == select(subquery.c.qa_id).where(
                        subquery.c.rn == qa_index_num + 1  # ROW_NUMBER从1开始
                    ).scalar_subquery()
                ).values(feedback=feedback)

                result = await session.execute(update_stmt)
                await session.commit()

                if result.rowcount > 0:
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
        获取会话中的最后一个QA对 - SQLAlchemy版本
        :param session_id: 会话ID
        :return: 最后一个QA对数据或None，包含agent_id
        """
        try:
            async with get_session() as session:
                # 获取最后一个QA记录
                stmt = select(KbotMdChatQa).where(
                    KbotMdChatQa.session_id == session_id
                ).order_by(
                    KbotMdChatQa.qa_id.desc()
                ).limit(1)

                result = await session.execute(stmt)
                last_session = result.scalar_one_or_none()

                if not last_session:
                    return None

                # 获取对应的参考文献
                ref_stmt = select(KbotMdChatReferences).where(
                    KbotMdChatReferences.qa_id == last_session.qa_id
                )
                ref_result = await session.execute(ref_stmt)
                references_rows = ref_result.scalars().all()

                # 转换向量数据
                qa_embedding = []
                if last_session.qa_embedding:
                    qa_embedding = OracleVecHandler().convert(vec=last_session.qa_embedding, to_string=False)
                    # 确保qa_embedding是Python列表
                    if hasattr(qa_embedding, 'tolist'):
                        qa_embedding = qa_embedding.tolist() # type: ignore
                    elif isinstance(qa_embedding, list):
                        qa_embedding = list(qa_embedding)

                # 构建参考文献列表
                references = []
                for ref_row in references_rows:
                    references.append(Reference(
                        chunk_type=ref_row.chunk_type,
                        chunk_file_path=ref_row.chunk_file_path,
                        page_num=ref_row.page_num,
                        content=safe_read_content(ref_row.chunk_content),
                        download_link=ref_row.download_link,
                        preview_link=ref_row.preview_link,
                        similarity_score=ref_row.similarity_score,
                        reranker_score=ref_row.reranker_score
                    ))

                # 将Reference对象转换为字典
                references_dict = [ref.to_dict() for ref in references]

                return {
                    "question": safe_read_content(last_session.question),
                    "answer": safe_read_content(last_session.answer),
                    "qa_embedding": qa_embedding,
                    "references": references_dict,
                    "feedback": last_session.feedback,
                    "by": last_session.username,
                    "request_time": last_session.request_time.strftime("%Y-%m-%d %H:%M:%S.%f") if last_session.request_time else None, # type: ignore
                    "response_time": last_session.response_time.strftime("%Y-%m-%d %H:%M:%S.%f") if last_session.response_time else None, # type: ignore
                    "agent_id": last_session.agent_id
                }

        except Exception as e:
            logger.error(f"获取最后一个QA数据失败: {e}")
            return None

    async def get_qa_data(self, session_id: str, qa_index_num: int) -> QAData | None:
        """
        获取单个QA对 - SQLAlchemy版本
        :param session_id: 会话ID
        :param qa_index_num: QA对索引
        :return: QA对数据或None
        """
        try:
            async with get_session() as session:
                # 使用窗口函数获取指定索引的QA记录
                subquery = select(
                    KbotMdChatQa.qa_id,
                    func.row_number().over(
                        order_by=KbotMdChatQa.request_time
                    ).label('rn')
                ).where(
                    KbotMdChatQa.session_id == session_id
                ).subquery()

                # 获取指定索引的QA记录
                stmt = select(KbotMdChatQa).where(
                    KbotMdChatQa.qa_id == select(subquery.c.qa_id).where(
                        subquery.c.rn == qa_index_num + 1  # ROW_NUMBER从1开始
                    ).scalar_subquery()
                )

                result = await session.execute(stmt)
                session_row = result.scalar_one_or_none()

                if not session_row:
                    return None

                # 获取对应的参考文献
                ref_stmt = select(KbotMdChatReferences).where(
                    KbotMdChatReferences.qa_id == session_row.qa_id
                )
                ref_result = await session.execute(ref_stmt)
                references_rows = ref_result.scalars().all()

                # 转换向量数据
                qa_embedding = []
                if session_row.qa_embedding:
                    qa_embedding = OracleVecHandler().convert(vec=session_row.qa_embedding, to_string=False)

                # 构建参考文献列表
                references = []
                for ref_row in references_rows:
                    references.append(Reference(
                        chunk_type=ref_row.chunk_type,
                        chunk_file_path=ref_row.chunk_file_path,
                        page_num=ref_row.page_num,
                        content=safe_read_content(ref_row.chunk_content),
                        download_link=ref_row.download_link,
                        preview_link=ref_row.preview_link,
                        similarity_score=ref_row.similarity_score,
                        reranker_score=ref_row.reranker_score
                    ))

                return QAData(
                    question=safe_read_content(session_row.question),
                    answer=safe_read_content(session_row.answer),
                    qa_embedding=qa_embedding, # type: ignore
                    references=references,
                    feedback=session_row.feedback,
                    by=session_row.username,
                    request_time=session_row.request_time.strftime("%Y-%m-%d %H:%M:%S.%f") if session_row.request_time else None, # type: ignore
                    response_time=session_row.response_time.strftime("%Y-%m-%d %H:%M:%S.%f") if session_row.response_time else None # type: ignore
                )

        except Exception as e:
            logger.error(f"获取QA数据失败: {e}")
            return None

    async def update_last_qa_data_answer(self, session_id: str, answer: str) -> bool:
        """
        更新会话中的最后一个QA对的答案 - SQLAlchemy版本
        :param session_id: 会话ID
        :param answer: 答案
        :return: 是否更新成功
        """
        try:
            async with get_session() as session:
                # 获取最后一个QA记录的qa_id
                subquery = select(KbotMdChatQa.qa_id).where(
                    KbotMdChatQa.session_id == session_id
                ).order_by(
                    KbotMdChatQa.request_time.desc()
                ).limit(1).scalar_subquery()

                # 更新答案
                update_stmt = update(KbotMdChatQa).where(
                    KbotMdChatQa.qa_id == subquery
                ).values(answer=answer)

                result = await session.execute(update_stmt)
                await session.commit()

                if result.rowcount > 0:
                    logger.debug(f"更新最后一个QA答案成功: {session_id}")
                    return True
                else:
                    logger.warning(f"未找到QA数据，无法更新答案: {session_id}")
                    return False

        except Exception as e:
            logger.error(f"更新最后一个QA答案失败: {e}")
            return False

    async def get_session_list(self, agent_id: int, page: int = 1, page_size: int = 10) -> tuple[list[dict], int]:
        """
        获取会话列表 - SQLAlchemy版本
        :param agent_id: agent ID
        :param page: 页码
        :param page_size: 每页大小
        :return: (会话列表, 总数量)
        """
        try:
            async with get_session() as session:
                # 获取唯一的会话ID列表
                distinct_stmt = select(KbotMdChatQa.session_id).where(
                    KbotMdChatQa.agent_id == agent_id
                ).distinct()

                # 计算总数
                count_stmt = select(func.count()).select_from(distinct_stmt.subquery())
                total_result = await session.execute(count_stmt)
                total_count = total_result.scalar_one()

                # 分页查询会话ID
                session_ids_stmt = distinct_stmt.order_by(
                    # 按最新会话时间排序
                    select(func.max(KbotMdChatQa.request_time)).where(
                        KbotMdChatQa.session_id == distinct_stmt.c.session_id
                    ).scalar_subquery().desc()
                ).offset((page - 1) * page_size).limit(page_size)

                session_ids_result = await session.execute(session_ids_stmt)
                session_ids = session_ids_result.scalars().all()

                # 获取每个会话的详细信息
                sessions_list = []
                for session_id in session_ids:
                    # 获取会话的基本信息（第一条记录）
                    first_qa_stmt = select(KbotMdChatQa).where(
                        KbotMdChatQa.session_id == session_id
                    ).order_by(KbotMdChatQa.request_time).limit(1)

                    first_qa_result = await session.execute(first_qa_stmt)
                    first_qa = first_qa_result.scalar_one()

                    # 获取QA总数
                    qa_count_stmt = select(func.count()).where(
                        KbotMdChatQa.session_id == session_id
                    )
                    qa_count_result = await session.execute(qa_count_stmt)
                    qa_count = qa_count_result.scalar_one()

                    sessions_list.append({
                        "session_id": session_id,
                        "agent_id": first_qa.agent_id,
                        "first_question": safe_read_content(first_qa.question),
                        "create_time": first_qa.request_time.strftime("%Y-%m-%d %H:%M:%S.%f") if first_qa.request_time else None, # type: ignore
                        "qa_count": qa_count
                    })

                return sessions_list, total_count

        except Exception as e:
            logger.error(f"获取会话列表失败: {e}")
            return [], 0