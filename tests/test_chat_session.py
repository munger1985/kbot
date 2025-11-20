import json
import array
import asyncio
import sys
import traceback
from pathlib import Path
from datetime import datetime
from decimal import Decimal
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# 加载环境变量
env_path = project_root / ".env"
load_dotenv(env_path)

from dao.repositories.kbot_md_chat_qa_repo import KbotMdChatQaRepository
from dao.entities.kbot_biz_chat_session import KbotBizChatSession, QAData, Reference
from utils.serializer import safe_serialize


repo = KbotMdChatQaRepository()

def create_sample_data():
    """创建符合要求的测试数据"""
    return KbotBizChatSession(
        session_id="session_" + str(datetime.now().timestamp()),
        agent_id=1,
        qa_data=[
            QAData(
                question="什么是人工智能?",
                answer="人工智能是模拟人类智能的计算机系统",
                qa_embedding=[0.1, 0.2, 0.3],
                references=[
                    Reference(
                        chunk_type=1,
                        chunk_file_path="",
                        page_num=1,
                        content="人工智能(AI)是指由机器展示的智能...",
                        download_link="https://example.com/ai.pdf",
                        preview_link="https://example.com/ai.pdf",
                        similarity_score=0.95,
                        reranker_score=0.88
                    )
                ],
                feedback=0,
                by="chris",
                request_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                response_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
        ]
    )

async def insert(key: str):
    """将数据写入Oracle"""
    try:
        data = create_sample_data()
        
        if repo is None:
            print("创建会话存储库实例失败")
            return False
        
        session_id = data.session_id
        
        # 将数据转为JSON字符串存储
        r = await repo.create_session(data)
        
        print(f"成功写入Oracle，session id: {session_id}")
        return r
    except Exception as e:
        print(f"写入Oracle失败: {str(e)}")
        return False

async def get_all(key: str):
    try:
        # 查询Oracle
        if repo is None:
            print("创建会话存储库实例失败")
            return False
        
        session_id = key
        print(f"正在查询Oracle，session_id: {session_id}")
        retrieved_data = await repo.get_session(session_id)
        
        if retrieved_data:
            print("\n从Oracle查询到的数据:")
            print(json.dumps(retrieved_data, indent=2, ensure_ascii=False, default=safe_serialize))
        else:
            print("查询Oracle失败，未找到数据")
            
    except Exception as e:
        print("\n查询Oracle失败，完整错误信息:")
        print(f"异常类型: {type(e).__name__}")
        print(f"异常消息: {str(e)}")
        print("\n堆栈跟踪:")
        print(traceback.format_exc())


async def insert_qa_data(key: str):
    # 创建测试数据
    data = QAData(
        question="Oracle是什么?",
        answer="Oracle是一个开源的内存数据结构存储",
        qa_embedding=[0.4, 0.5, 0.6],
        references=[
            Reference(
                chunk_type=2,
                chunk_file_path="/path/to/Oracle.pdf",
                page_num=2,
                content="Oracle是一个开源的、支持网络、基于内存...",
                download_link="https://example.com/Oracle.pdf",
                preview_link="https://example.com/Oracle.pdf",
                similarity_score=0.92,
                reranker_score=0.85
            )
        ],
        feedback=1,
        by="chris",
        request_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        response_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    
    # 写入Oracle
    if repo is None:
        print("创建会话存储库实例失败")
        return False
    result = await repo.add_qa_data(key, data)
    if result:
        print("写入Oracle成功")
    else:
        print("写入Oracle失败")

async def get_qa_data(key: str):
    if repo is None:
        print("创建会话存储库实例失败")
        return False
    
    result = await repo.get_qa_data(key, 0)
    if result:
        print("\n从Oracle查询到的数据:")
        # 使用安全的序列化方式
        def safe_serialize(obj):
            if hasattr(obj, 'to_dict'):
                return obj.to_dict()
            elif isinstance(obj, (str, int, float, bool)):
                return obj
            elif isinstance(obj, Decimal):
                return float(obj)
            elif isinstance(obj, array.array):
                return list(obj)
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            else:
                return str(obj)
        
        print(json.dumps(result, indent=2, ensure_ascii=False, default=safe_serialize))
    else:
        print("查询Oracle失败，未找到数据")

async def delete_session(key: str):
    if repo is None:
        print("创建会话存储库实例失败")
        return False
    result = await repo.delete_session(key)
    if result:
        print("删除Oracle成功")
    else:
        print("删除Oracle失败")

async def update_qa_feedback(key: str):
    try:
        if repo is None:
            print("创建会话存储库实例失败")
            return False
        print(f"正在写入Oracle，session_id: {key}")
        result = await repo.update_qa_feedback(key, 0, -3)
        if result:
            print("写入Oracle成功")
            await get_qa_data(key)
        else:
            print("写入Oracle失败")
    except Exception as e:
        print("\n写入Oracle失败，完整错误信息:")
        print(f"异常类型: {type(e).__name__}")
        print(f"异常消息: {str(e)}")
        print("\n堆栈跟踪:")
        print(traceback.format_exc())

async def get_last_qa_data(key: str):
    try:
        if repo is None:
            print("创建会话存储库实例失败")
            return False
        print(f"正在查询Oracle，session_id: {key}")
        result = await repo.get_last_qa_data(key)
        if result:
            print("\n从Oracle查询到的数据:")
            # 使用安全的序列化方式
            def safe_serialize(obj):
                if isinstance(obj, (str, int, float, bool)):
                    return obj
                elif isinstance(obj, Decimal):
                    return float(obj)
                elif isinstance(obj, array.array):
                    return list(obj)
                elif isinstance(obj, datetime):
                    return obj.isoformat()
                elif hasattr(obj, '__dict__'):
                    return obj.__dict__
                else:
                    return str(obj)
            
            print(json.dumps(result, indent=2, ensure_ascii=False, default=safe_serialize))
        else:
            print("查询Oracle失败，未找到数据")
    except Exception as e:
        print("\n查询Oracle失败，完整错误信息:")
        print(f"异常类型: {type(e).__name__}")
        print(f"异常消息: {str(e)}")
        print("\n堆栈跟踪:")
        print(traceback.format_exc())

async def update_last_qa_data_answer(key: str):
    try:
        if repo is None:
            print("创建会话存储库实例失败")
            return False
        print(f"正在写入Oracle，session_id: {key}")
        result = await repo.update_last_qa_data_answer(key, "您好！请问有什么可以帮助您的吗？")
        if result:
            print("写入Oracle成功")
            await get_last_qa_data(key)
        else:
            print("写入Oracle失败")
    except Exception as e:
        print("\n查询Oracle失败，完整错误信息:")
        print(f"异常类型: {type(e).__name__}")
        print(f"异常消息: {str(e)}")
        print("\n堆栈跟踪:")
        print(traceback.format_exc())

async def delete_by_agent(agent_id: int):
    if repo is None:
        print("创建会话存储库实例失败")
        return False
    result = await repo.delete_by_agent_id(agent_id)
    if result:
        print("删除Oracle成功")
    else:
        print("删除Oracle失败")

async def main():
    """主测试函数，确保连接正确关闭"""
    print("Starting Oracle test...")
    key = "session_1763610729.431471"
    kb_id = 66
    agent_id = 1
    # 执行测试
    # await insert(key)
    await get_all(key)
    # await insert_qa_data(key)
    # await get_qa_data(key)
    # await update_qa_feedback(key)
    # await update_last_qa_data_answer(key)
    # await get_last_qa_data(key)
    # await delete_session(key)
    # await delete_by_agent(agent_id)
    print("Oracle test finished.")

if __name__ == "__main__":
    asyncio.run(main())