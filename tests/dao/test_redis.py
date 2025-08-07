import json
import asyncio
import sys
import traceback
from pathlib import Path
from datetime import datetime
import time

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from dao.repositories.kbot_md_chat_session_repo import KbotMdChatSessionRepository

key="session_1754552000.5921"



def create_sample_data():
    """创建符合要求的测试数据"""
    return {
        "SESSION_ID": "session_" + str(datetime.now().timestamp()),
        "AGENT_ID": 1,
        "QA_DATA": [
            {
                "question": "什么是人工智能?",
                "answer": "人工智能是模拟人类智能的计算机系统",
                "qa_embedding": "",
                "references": [
                    {
                        "chunk_type": 1,
                        "chunk_file_path": "",
                        "file_ext": ".pdf",
                        "page_num": 1,
                        "content": "人工智能(AI)是指由机器展示的智能...",
                        "download_link": "https://example.com/ai.pdf",
                        "preview_link": "https://example.com/ai.pdf",
                        "similarity_score": 0.95,
                        "reranker_score": 0.88
                    }
                ],
                "feedback": 0,
                "by": "chris",
                "request_time": "2025-07-31 12:00:00",
                "response_time": "2025-07-31 12:00:01"
            }
        ]
    }

async def insert():
    """将数据写入Redis"""
    try:
        data = create_sample_data()
        repo = KbotMdChatSessionRepository()
        session_id = data['SESSION_ID']
        
        # 将数据转为JSON字符串存储
        r = await repo.create_session(data)
        

        
        print(f"成功写入Redis，session id: {session_id}")
        return r
    except Exception as e:
        print(f"写入Redis失败: {str(e)}")
        return False

async def get_all():
    try:
        # 查询Redis
        repo = KbotMdChatSessionRepository()
        session_id = key
        print(f"正在查询Redis，session_id: {session_id}")
        retrieved_data = await repo.get_session(session_id)
        
        if retrieved_data:
            print("\n从Redis查询到的数据:")
            print(json.dumps(retrieved_data, indent=2, ensure_ascii=False))
        else:
            print("查询Redis失败，未找到数据")
    except Exception as e:
        print("\n查询Redis失败，完整错误信息:")
        print(f"异常类型: {type(e).__name__}")
        print(f"异常消息: {str(e)}")
        print("\n堆栈跟踪:")
        print(traceback.format_exc())

async def insert_qa_data():
    # 创建测试数据
    data = """{
                "question": "Redis是什么?",
                "answer": "Redis是一个开源的内存数据结构存储",
                "qa_embedding": "329195767893316712",
                "references": [
                    {
                        "chunk_type": 2,
                        "chunk_file_path": "",
                        "file_ext": ".pdf",
                        "page_num": 2,
                        "content": "Redis是一个开源的、支持网络、基于内存...",
                        "download_link": "https://example.com/redis.pdf",
                        "preview_link": "https://example.com/redis.pdf",
                        "similarity_score": 0.92,
                        "reranker_score": 0.85
                    }
                ],
                "feedback": 1,
                "by": "chris",
                "request_time": "2025-07-31 12:00:00",
                "response_time": "2025-07-31 12:00:01"
            }"""
    sample_data = json.loads(data)
    # 写入Redis
    repo = KbotMdChatSessionRepository()
    result = await repo.add_qa_data(key, sample_data)
    if result:
        print("写入Redis成功")
    else:
        print("写入Redis失败")

async def get_qa_data():

    repo = KbotMdChatSessionRepository()
    result = await repo.get_qa_data(key, 0)
    if result:
        print("\n从Redis查询到的数据:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("查询Redis失败，未找到数据")
async def delete_session():
    repo = KbotMdChatSessionRepository()
    result = await repo.delete_session(key)
    if result:
        print("删除Redis成功")
    else:
        print("删除Redis失败")

async def update_qa_feedback():
    try:
        repo = KbotMdChatSessionRepository()
        print(f"正在写入Redis，session_id: {key}")
        result = await repo.update_qa_feedback(key, 0, -1)
        if result:
            print("写入Redis成功")
            await get_qa_data()
        else:
            print("写入Redis失败")
    except Exception as e:
        print("\n写入Redis失败，完整错误信息:")
        print(f"异常类型: {type(e).__name__}")
        print(f"异常消息: {str(e)}")
        print("\n堆栈跟踪:")
        print(traceback.format_exc())

async def get_last_qa_data():
    try:
        repo = KbotMdChatSessionRepository()
        print(f"正在查询Redis，session_id: {key}")
        result = await repo.get_last_qa_data(key)
        if result:
            print("\n从Redis查询到的数据:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("查询Redis失败，未找到数据")
    except Exception as e:
        print("\n查询Redis失败，完整错误信息:")
        print(f"异常类型: {type(e).__name__}")
        print(f"异常消息: {str(e)}")
        print("\n堆栈跟踪:")
        print(traceback.format_exc())

async def update_last_qa_data_answer():
    try:
        repo = KbotMdChatSessionRepository()
        print(f"正在写入Redis，session_id: {key}")
        result = await repo.update_last_qa_data_answer(key, "Redis啊redis")
        if result:
            print("写入Redis成功")
            await get_last_qa_data()
        else:
            print("写入Redis失败")
    except Exception as e:
        print("\n查询Redis失败，完整错误信息:")
        print(f"异常类型: {type(e).__name__}")
        print(f"异常消息: {str(e)}")
        print("\n堆栈跟踪:")
        print(traceback.format_exc())

if __name__ == "__main__":
    print("Starting redis test...")

    asyncio.run(get_all())

    
    