import json
import asyncio
import sys
import traceback
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# 加载环境变量
env_path = project_root / ".env"
load_dotenv(env_path)

from dao.repositories.kbot_biz_chat_session_factory import ChatSessionRepositoryFactory
from dao.entities.kbot_biz_chat_session import KbotBizChatSession, QAData, Reference

key="session_1762925678.397321"
kb_id=66

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
                        file_ext=".pdf",
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
                request_time="2025-07-31 12:00:00",
                response_time="2025-07-31 12:00:01"
            )
        ]
    )

async def insert():
    """将数据写入ElasticSearch"""
    try:
        data = create_sample_data()
        repo = await ChatSessionRepositoryFactory.create_repository(kb_id)
        if repo is None:
            print("创建会话存储库实例失败")
            return False
        
        session_id = data.session_id
        
        # 将数据转为JSON字符串存储
        r = await repo.create_session(data)
        

        
        print(f"成功写入ElasticSearch，session id: {session_id}")
        return r
    except Exception as e:
        print(f"写入ElasticSearch失败: {str(e)}")
        return False

async def get_all():
    try:
        # 查询ElasticSearch
        repo = await ChatSessionRepositoryFactory.create_repository(kb_id)
        if repo is None:
            print("创建会话存储库实例失败")
            return False
        
        session_id = key
        print(f"正在查询ElasticSearch，session_id: {session_id}")
        retrieved_data = await repo.get_session(session_id)
        
        if retrieved_data:
            print("\n从ElasticSearch查询到的数据:")
            # 检查返回的数据类型，如果是字典直接打印，如果是对象则调用to_dict()
            if hasattr(retrieved_data, 'to_dict'):
                print(json.dumps(retrieved_data.to_dict(), indent=2, ensure_ascii=False))
            else:
                print(json.dumps(retrieved_data, indent=2, ensure_ascii=False))
        else:
            print("查询ElasticSearch失败，未找到数据")
    except Exception as e:
        print("\n查询ElasticSearch失败，完整错误信息:")
        print(f"异常类型: {type(e).__name__}")
        print(f"异常消息: {str(e)}")
        print("\n堆栈跟踪:")
        print(traceback.format_exc())

async def insert_qa_data():
    # 创建测试数据
    data = QAData(
                question="ElasticSearch是什么?",
                answer="ElasticSearch是一个开源的内存数据结构存储",
                qa_embedding=[0.4, 0.5, 0.6],
                references=[
                    Reference(
                        chunk_type=2,
                        chunk_file_path="",
                        file_ext=".pdf",
                        page_num=2,
                        content="ElasticSearch是一个开源的、支持网络、基于内存...",
                        download_link="https://example.com/ElasticSearch.pdf",
                        preview_link="https://example.com/ElasticSearch.pdf",
                        similarity_score=0.92,
                        reranker_score=0.85
                    )
                ],
                feedback=1,
                by="chris",
                request_time="2025-07-31 12:00:00",
                response_time="2025-07-31 12:00:01"
    )
     # 写入ElasticSearch
    repo = await ChatSessionRepositoryFactory.create_repository(kb_id)
    if repo is None:
        print("创建会话存储库实例失败")
        return False
    result = await repo.add_qa_data(key, data)
    if result:
        print("写入ElasticSearch成功")
    else:
        print("写入ElasticSearch失败")

async def get_qa_data():

    repo = await ChatSessionRepositoryFactory.create_repository(kb_id)
    if repo is None:
        print("创建会话存储库实例失败")
        return False
    
    result = await repo.get_qa_data(key, 0)
    if result:
        print("\n从ElasticSearch查询到的数据:")
        # 检查返回的数据类型，如果是字典直接打印，如果是对象则调用to_dict()
        if hasattr(result, 'to_dict'):
            print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("查询ElasticSearch失败，未找到数据")

async def delete_session():
    repo = await ChatSessionRepositoryFactory.create_repository(kb_id)
    if repo is None:
        print("创建会话存储库实例失败")
        return False
    result = await repo.delete_session(key)
    if result:
        print("删除ElasticSearch成功")
    else:
        print("删除ElasticSearch失败")

async def update_qa_feedback():
    try:
        repo = await ChatSessionRepositoryFactory.create_repository(kb_id)
        if repo is None:
            print("创建会话存储库实例失败")
            return False
        print(f"正在写入ElasticSearch，session_id: {key}")
        result = await repo.update_qa_feedback(key, 0, -3)
        if result:
            print("写入ElasticSearch成功")
            await get_qa_data()
        else:
            print("写入ElasticSearch失败")
    except Exception as e:
        print("\n写入ElasticSearch失败，完整错误信息:")
        print(f"异常类型: {type(e).__name__}")
        print(f"异常消息: {str(e)}")
        print("\n堆栈跟踪:")
        print(traceback.format_exc())

async def get_last_qa_data():
    try:
        repo = await ChatSessionRepositoryFactory.create_repository(kb_id)
        if repo is None:
            print("创建会话存储库实例失败")
            return False
        print(f"正在查询ElasticSearch，session_id: {key}")
        result = await repo.get_last_qa_data(key)
        if result:
            print("\n从ElasticSearch查询到的数据:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("查询ElasticSearch失败，未找到数据")
    except Exception as e:
        print("\n查询ElasticSearch失败，完整错误信息:")
        print(f"异常类型: {type(e).__name__}")
        print(f"异常消息: {str(e)}")
        print("\n堆栈跟踪:")
        print(traceback.format_exc())

async def update_last_qa_data_answer():
    try:
        repo = await ChatSessionRepositoryFactory.create_repository(kb_id)
        if repo is None:
            print("创建会话存储库实例失败")
            return False
        print(f"正在写入ElasticSearch，session_id: {key}")
        result = await repo.update_last_qa_data_answer(key, "您好！请问有什么可以帮助您的吗？")
        if result:
            print("写入ElasticSearch成功")
            await get_last_qa_data()
        else:
            print("写入ElasticSearch失败")
    except Exception as e:
        print("\n查询ElasticSearch失败，完整错误信息:")
        print(f"异常类型: {type(e).__name__}")
        print(f"异常消息: {str(e)}")
        print("\n堆栈跟踪:")
        print(traceback.format_exc())

async def delete_by_agent():
    repo = await ChatSessionRepositoryFactory.create_repository(kb_id)
    if repo is None:
        print("创建会话存储库实例失败")
        return False
    result = await repo.delete_by_agent_id(1)
    if result:
        print("删除ElasticSearch成功")
    else:
        print("删除ElasticSearch失败")

async def main():
    """主测试函数，确保连接正确关闭"""
    try:
        print("Starting ElasticSearch test...")
        # 执行测试
        # await insert()
        # await get_all()
        # await insert_qa_data()
        # await get_qa_data()
        # await update_qa_feedback()
        # await update_last_qa_data_answer()
        # await get_last_qa_data()
        # await delete_session()
        await delete_by_agent()
        print("ElasticSearch test finished.")
    finally:
        # 确保关闭所有连接
        from core.database.vec_elasticsearch import es_client_manager
        await es_client_manager.close_all()

if __name__ == "__main__":
    asyncio.run(main())


