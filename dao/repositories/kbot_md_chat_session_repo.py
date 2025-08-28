# import json
# from core.database.meta_redis import AsyncRedisPool

# class KbotMdChatSessionRepository:
#     def __init__(self):
#         """
#         初始化会话仓库，内部创建 AsyncRedisPool 实例
#         """
#         self.redis = AsyncRedisPool(db=0) # 0 为聊天session数据库

#     async def close(self) -> None:
#         """
#         关闭Redis连接
#         """
#         await self.redis.close()

#     async def create_session(self, session_data: dict) -> bool:
#         """
#         创建新会话
#         :param session_data: 会话数据
#         :return: 是否创建成功
#         """
#         session_id = session_data["session_id"]
#         session_key = f"session:{session_id}"
#         qa_data_key = f"{session_key}:qa_data"

#         async with self.redis as redis:
#             # 存储基本信息
#             await redis.execute_command(
#                 "HSET", session_key,
#                 "agent_id", session_data["agent_id"]
#             )

#             # 存储QA对
#             qa_data = session_data["qa_data"][0]
#             qa_key = f"qa:{session_id}:0"

#             # 存储QA基本信息
#             await redis.execute_command(
#                 "HSET", qa_key,
#                 "question", qa_data["question"],
#                 "answer", qa_data["answer"],
#                 "qa_embedding", qa_data["qa_embedding"],
#                 "feedback", qa_data["feedback"],
#                 "by", qa_data["by"],
#                 "request_time", qa_data["request_time"],
#                 "response_time", qa_data["response_time"]
#             )
            
#             # 存储参考资料
#             refs_key = f"{qa_key}:references"
#             for ref in qa_data["references"]:
#                 await redis.execute_command("RPUSH", refs_key, json.dumps(ref))
            
#             # 添加到QA对集合
#             await redis.execute_command("ZADD", qa_data_key, 0, qa_key)
            
#             return True

#     async def get_session(self, session_id: str) -> dict | None:
#         """
#         获取完整会话数据
#         :param session_id: 会话ID
#         :return: 会话数据或None
#         """
#         session_key = f"session:{session_id}"
#         qa_data_key = f"{session_key}:qa_data"

#         async with self.redis as redis:
#             # 检查会话是否存在
#             if not await redis.exists(session_key):
#                 return None

#             # 获取基本信息
#             basic_info = await redis.hgetall(session_key)
            
#             # 获取所有QA对键
#             qa_keys = await redis.execute_command("ZRANGE", qa_data_key, 0, -1)
            
#             qa_data = []
#             for qa_key in qa_keys:
#                 # 获取QA基本信息
#                 qa_info = await redis.hgetall(qa_key)
                
#                 # 获取参考资料
#                 refs_key = f"{qa_key}:references"
#                 references = await redis.execute_command("LRANGE", refs_key, 0, -1)
                
#                 qa_data.append({
#                     "question": qa_info["question"],
#                     "answer": qa_info["answer"],
#                     "qa_embedding": qa_info["qa_embedding"],
#                     "references": [json.loads(ref) for ref in references],
#                     "feedback": int(qa_info["feedback"]),
#                     "by": qa_info["by"],
#                     "request_time": qa_info["request_time"],
#                     "response_time": qa_info["response_time"]
#                 })
            
#             return {
#                 "session_id": session_id,
#                 "agent_id": int(basic_info["agent_id"]),
#                 "qa_data": qa_data
#             }

#     async def add_qa_data(self, session_id: str, qa_data: dict) -> bool:
#         """
#         向会话添加新的QA对
#         :param session_id: 会话ID
#         :param qa_data: QA对数据
#         :return: 是否添加成功
#         """
#         session_key = f"session:{session_id}"
#         qa_data_key = f"{session_key}:qa_data"

#         async with self.redis as redis:
#             # 检查会话是否存在
#             if not await redis.exists(session_key):
#                 return False

#             # 获取当前QA对数量作为新索引
#             count = await redis.execute_command("ZCARD", qa_data_key)
#             qa_key = f"qa:{session_id}:{count}"
            
#             # 存储QA基本信息
#             await redis.execute_command(
#                 "HSET", qa_key,
#                 "question", qa_data["question"],
#                 "answer", qa_data["answer"],
#                 "qa_embedding", qa_data["qa_embedding"],
#                 "feedback", qa_data["feedback"],
#                 "by", qa_data["by"],
#                 "request_time", qa_data["request_time"],
#                 "response_time", qa_data["response_time"]
#             )
            
#             # 存储参考资料
#             refs_key = f"{qa_key}:references"
#             for ref in qa_data["references"]:
#                 await redis.execute_command("RPUSH", refs_key, json.dumps(ref))
            
#             # 添加到QA对集合
#             await redis.execute_command("ZADD", qa_data_key, count, qa_key)
            
#             return True

#     async def delete_session(self, session_id: str) -> bool:
#         """
#         删除整个会话
#         :param session_id: 会话ID
#         :return: 是否删除成功
#         """
#         session_key = f"session:{session_id}"
#         qa_data_key = f"{session_key}:qa_data"

#         async with self.redis as redis:
#             # 获取所有QA对键以便删除
#             qa_keys = await redis.execute_command("ZRANGE", qa_data_key, 0, -1)
            
#             # 删除所有QA对和相关参考资料
#             for qa_key in qa_keys:
#                 refs_key = f"{qa_key}:references"
#                 await redis.delete(qa_key, refs_key)
            
#             # 删除主键和QA对集合
#             await redis.delete(session_key, qa_data_key)
            
#             return True

#     async def update_qa_feedback(self, session_id: str, qa_index: int, feedback: int) -> bool:
#         """
#         更新QA对的反馈标记
#         :param session_id: 会话ID
#         :param qa_index: QA对索引
#         :param feedback: 反馈值
#         :return: 是否更新成功
#         """
#         qa_key = f"qa:{session_id}:{qa_index}"

#         async with self.redis as redis:
#             await redis.execute_command("HSET", qa_key, "feedback", feedback)
#             return True

#     async def get_last_qa_data(self, session_id: str) -> dict | None:
#         """
#         获取会话中的最后一个QA对
#         :param session_id: 会话ID
#         :return: 最后一个QA对数据或None，包含agent_id
#         """
#         session_key = f"session:{session_id}"
#         qa_data_key = f"{session_key}:qa_data"
        
#         async with self.redis as redis:
#             # 获取session信息
#             session_info = await redis.hgetall(session_key)
#             if not session_info:
#                 return None
                
#             # 获取最后一个QA对
#             qa_keys = await redis.execute_command("ZRANGE", qa_data_key, -1, -1)
#             if not qa_keys:
#                 return None
#             qa_key = qa_keys[0]
            
#             refs_key = f"{qa_key}:references"
#             qa_info = await redis.hgetall(qa_key)
#             references = await redis.execute_command("LRANGE", refs_key, 0, -1)
            
#             return {
#                 "question": qa_info["question"],
#                 "answer": qa_info["answer"],
#                 "qa_embedding": qa_info["qa_embedding"],
#                 "references": [json.loads(ref) for ref in references],
#                 "feedback": int(qa_info["feedback"]),
#                 "by": qa_info["by"],
#                 "request_time": qa_info["request_time"],
#                 "response_time": qa_info["response_time"],
#                 "agent_id": int(session_info["agent_id"])
#             }

#     async def get_qa_data(self, session_id: str, qa_index: int) -> dict | None:
#         """
#         获取单个QA对
#         :param session_id: 会话ID
#         :param qa_index: QA对索引
#         :return: QA对数据或None
#         """
#         qa_key = f"qa:{session_id}:{qa_index}"
#         refs_key = f"{qa_key}:references"
        
#         async with self.redis as redis:
#             if not await redis.exists(qa_key):
#                 return None
            
#             qa_info = await redis.hgetall(qa_key)
#             references = await redis.execute_command("LRANGE", refs_key, 0, -1)
            
#             return {
#                 "question": qa_info["question"],
#                 "answer": qa_info["answer"],
#                 "qa_embedding": qa_info["qa_embedding"],
#                 "references": [json.loads(ref) for ref in references],
#                 "feedback": int(qa_info["feedback"]),
#                 "by": qa_info["by"],
#                 "request_time": qa_info["request_time"],
#                 "response_time": qa_info["response_time"]
#             }

    
#     async def update_last_qa_data_answer(self, session_id: str, answer: str) -> bool:
#         """
#         更新会话中的最后一个QA对的答案
#         :param session_id: 会话ID
#         :param answer: 答案
#         :return: 是否更新成功
#         """
#         session_key = f"session:{session_id}"
#         qa_data_key = f"{session_key}:qa_data"
        
#         async with self.redis as redis:
#             # 获取最后一个QA对的键
#             qa_keys = await redis.execute_command("ZRANGE", qa_data_key, -1, -1)
#             if not qa_keys:
#                 return False
#             qa_key = qa_keys[0]
            
#             # 更新答案
#             await redis.execute_command("HSET", qa_key, "answer", answer)
#             return True

import json
from typing import Dict, List, Optional
from core.database.meta_redis import AsyncRedisPool

class KbotMdChatSessionRepository:
    def __init__(self):
        """
        初始化会话仓库，内部创建 AsyncRedisPool 实例
        """
        self.redis = AsyncRedisPool(db=0)  # 0 为聊天session数据库

    async def close(self) -> None:
        """
        关闭Redis连接
        """
        await self.redis.close()

    async def create_session(self, session_data: dict) -> bool:
        """
        创建新会话
        :param session_data: 会话数据
        :return: 是否创建成功
        """
        session_id = session_data["session_id"]
        session_key = f"session:{session_id}"
        qa_data_key = f"{session_key}:qa_data"

        async with self.redis as redis:
            # 存储基本信息
            await redis.hset(session_key, mapping={
                "agent_id": session_data["agent_id"]
            })

            # 存储QA对
            qa_data = session_data["qa_data"][0]
            qa_key = f"qa:{session_id}:0"

            # 存储QA基本信息
            await redis.hset(qa_key, mapping={
                "question": qa_data["question"],
                "answer": qa_data["answer"],
                "qa_embedding": qa_data["qa_embedding"],
                "feedback": qa_data["feedback"],
                "by": qa_data["by"],
                "request_time": qa_data["request_time"],
                "response_time": qa_data["response_time"]
            })
            
            # 存储参考资料
            refs_key = f"{qa_key}:references"
            for ref in qa_data["references"]:
                await redis.rpush(refs_key, json.dumps(ref))
            
            # 添加到QA对集合
            await redis.zadd(qa_data_key, {qa_key: 0})
            
            return True

    async def get_session(self, session_id: str) -> Optional[Dict]:
        """
        获取完整会话数据
        :param session_id: 会话ID
        :return: 会话数据或None
        """
        session_key = f"session:{session_id}"
        qa_data_key = f"{session_key}:qa_data"

        async with self.redis as redis:
            # 检查会话是否存在
            if not await redis.exists(session_key):
                return None

            # 获取基本信息
            basic_info = await redis.hgetall(session_key)
            
            # 获取所有QA对键
            qa_keys = await redis.zrange(qa_data_key, 0, -1)
            
            qa_data = []
            for qa_key in qa_keys:
                # 获取QA基本信息
                qa_info = await redis.hgetall(qa_key)
                
                # 获取参考资料
                refs_key = f"{qa_key}:references"
                references = await redis.lrange(refs_key, 0, -1)
                
                qa_data.append({
                    "question": qa_info["question"],
                    "answer": qa_info["answer"],
                    "qa_embedding": qa_info["qa_embedding"],
                    "references": [json.loads(ref) for ref in references],
                    "feedback": int(qa_info["feedback"]),
                    "by": qa_info["by"],
                    "request_time": qa_info["request_time"],
                    "response_time": qa_info["response_time"]
                })
            
            return {
                "session_id": session_id,
                "agent_id": int(basic_info["agent_id"]),
                "qa_data": qa_data
            }

    async def add_qa_data(self, session_id: str, qa_data: dict) -> bool:
        """
        向会话添加新的QA对
        :param session_id: 会话ID
        :param qa_data: QA对数据
        :return: 是否添加成功
        """
        session_key = f"session:{session_id}"
        qa_data_key = f"{session_key}:qa_data"

        async with self.redis as redis:
            # 检查会话是否存在
            if not await redis.exists(session_key):
                return False

            # 获取当前QA对数量作为新索引
            count = await redis.zcard(qa_data_key)
            qa_key = f"qa:{session_id}:{count}"
            
            # 存储QA基本信息
            await redis.hset(qa_key, mapping={
                "question": qa_data["question"],
                "answer": qa_data["answer"],
                "qa_embedding": qa_data["qa_embedding"],
                "feedback": qa_data["feedback"],
                "by": qa_data["by"],
                "request_time": qa_data["request_time"],
                "response_time": qa_data["response_time"]
            })
            
            # 存储参考资料
            refs_key = f"{qa_key}:references"
            for ref in qa_data["references"]:
                await redis.rpush(refs_key, json.dumps(ref))
            
            # 添加到QA对集合
            await redis.zadd(qa_data_key, {qa_key: count})
            
            return True

    async def delete_session(self, session_id: str) -> bool:
        """
        删除整个会话
        :param session_id: 会话ID
        :return: 是否删除成功
        """
        session_key = f"session:{session_id}"
        qa_data_key = f"{session_key}:qa_data"

        async with self.redis as redis:
            # 获取所有QA对键以便删除
            qa_keys = await redis.zrange(qa_data_key, 0, -1)
            
            # 删除所有QA对和相关参考资料
            for qa_key in qa_keys:
                refs_key = f"{qa_key}:references"
                await redis.delete(qa_key, refs_key)
            
            # 删除主键和QA对集合
            await redis.delete(session_key, qa_data_key)
            
            return True

    async def update_qa_feedback(self, session_id: str, qa_index: int, feedback: int) -> bool:
        """
        更新QA对的反馈标记
        :param session_id: 会话ID
        :param qa_index: QA对索引
        :param feedback: 反馈值
        :return: 是否更新成功
        """
        qa_key = f"qa:{session_id}:{qa_index}"

        async with self.redis as redis:
            await redis.hset(qa_key, {"feedback": feedback})
            return True

    async def get_last_qa_data(self, session_id: str) -> Optional[Dict]:
        """
        获取会话中的最后一个QA对
        :param session_id: 会话ID
        :return: 最后一个QA对数据或None，包含agent_id
        """
        session_key = f"session:{session_id}"
        qa_data_key = f"{session_key}:qa_data"
        
        async with self.redis as redis:
            # 获取session信息
            session_info = await redis.hgetall(session_key)
            if not session_info:
                return None
                
            # 获取最后一个QA对
            qa_keys = await redis.zrange(qa_data_key, -1, -1)
            if not qa_keys:
                return None
            qa_key = qa_keys[0]
            
            refs_key = f"{qa_key}:references"
            qa_info = await redis.hgetall(qa_key)
            references = await redis.lrange(refs_key, 0, -1)
            
            return {
                "question": qa_info["question"],
                "answer": qa_info["answer"],
                "qa_embedding": qa_info["qa_embedding"],
                "references": [json.loads(ref) for ref in references],
                "feedback": int(qa_info["feedback"]),
                "by": qa_info["by"],
                "request_time": qa_info["request_time"],
                "response_time": qa_info["response_time"],
                "agent_id": int(session_info["agent_id"])
            }

    async def get_qa_data(self, session_id: str, qa_index: int) -> Optional[Dict]:
        """
        获取单个QA对
        :param session_id: 会话ID
        :param qa_index: QA对索引
        :return: QA对数据或None
        """
        qa_key = f"qa:{session_id}:{qa_index}"
        refs_key = f"{qa_key}:references"
        
        async with self.redis as redis:
            if not await redis.exists(qa_key):
                return None
            
            qa_info = await redis.hgetall(qa_key)
            references = await redis.lrange(refs_key, 0, -1)
            
            return {
                "question": qa_info["question"],
                "answer": qa_info["answer"],
                "qa_embedding": qa_info["qa_embedding"],
                "references": [json.loads(ref) for ref in references],
                "feedback": int(qa_info["feedback"]),
                "by": qa_info["by"],
                "request_time": qa_info["request_time"],
                "response_time": qa_info["response_time"]
            }

    async def update_last_qa_data_answer(self, session_id: str, answer: str) -> bool:
        """
        更新会话中的最后一个QA对的答案
        :param session_id: 会话ID
        :param answer: 答案
        :return: 是否更新成功
        """
        session_key = f"session:{session_id}"
        qa_data_key = f"{session_key}:qa_data"
        
        async with self.redis as redis:
            # 获取最后一个QA对的键
            qa_keys = await redis.zrange(qa_data_key, -1, -1)
            if not qa_keys:
                return False
            qa_key = qa_keys[0]
            
            # 更新答案
            await redis.hset(qa_key, {"answer":answer})
            return True