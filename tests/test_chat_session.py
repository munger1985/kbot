import json
import array
import decimal
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

from dao.repositories.kbot_md_chat_session_repo import KbotMdChatSessionRepository
from dao.entities.kbot_biz_chat_session import KbotBizChatSession, QAData, Reference

key = "session_1763020747.04609"
kb_id = 66
repo = KbotMdChatSessionRepository()

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

async def get_all():
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
            
            # 增强的序列化函数 - 处理所有常见的数据类型
            def serialize_chat_session(obj):
                # 处理 Decimal 类型
                if isinstance(obj, Decimal):
                    return float(obj)
                
                # 处理 array 类型
                if isinstance(obj, array.array):
                    return list(obj)
                
                # 处理 datetime 类型
                if isinstance(obj, datetime):
                    return obj.isoformat()
                
                # 处理其他数值类型
                if isinstance(obj, (int, float)):
                    return obj
                
                # 处理 KbotMdChatSession 对象
                if hasattr(obj, '__class__') and obj.__class__.__name__ == 'KbotMdChatSession':
                    result = {
                        'session_id': obj.session_id,
                        'agent_id': int(obj.agent_id) if obj.agent_id else None,
                        'qa_data': []
                    }
                    
                    if hasattr(obj, 'qa_data') and obj.qa_data:
                        for qa in obj.qa_data:
                            qa_dict = {
                                'question': safe_convert(qa.question),
                                'answer': safe_convert(qa.answer),
                                'qa_embedding': safe_convert(qa.qa_embedding),
                                'feedback': safe_convert(qa.feedback),
                                'by': safe_convert(qa.by),
                                'request_time': safe_convert(qa.request_time),
                                'response_time': safe_convert(qa.response_time),
                                'references': []
                            }
                            
                            if hasattr(qa, 'references') and qa.references:
                                for ref in qa.references:
                                    ref_dict = {
                                        'chunk_type': safe_convert(ref.chunk_type),
                                        'chunk_file_path': safe_convert(ref.chunk_file_path),
                                        'file_ext': safe_convert(ref.file_ext),
                                        'page_num': safe_convert(ref.page_num),
                                        'content': safe_convert(ref.content),
                                        'download_link': safe_convert(ref.download_link),
                                        'preview_link': safe_convert(ref.preview_link),
                                        'similarity_score': safe_convert(ref.similarity_score),
                                        'reranker_score': safe_convert(ref.reranker_score)
                                    }
                                    qa_dict['references'].append(ref_dict)
                            
                            result['qa_data'].append(qa_dict)
                    return result
                
                # 处理其他无法序列化的类型
                return str(obj)
            
            # 辅助函数用于安全转换各种数据类型
            def safe_convert(value):
                if value is None:
                    return None
                elif isinstance(value, (str, int, float, bool)):
                    return value
                elif isinstance(value, Decimal):
                    return float(value)
                elif isinstance(value, array.array):
                    return list(value)
                elif isinstance(value, datetime):
                    return value.isoformat()
                else:
                    return str(value)
            
            # 使用安全的序列化方式
            serialized_data = serialize_chat_session(retrieved_data)
            print(json.dumps(serialized_data, indent=2, ensure_ascii=False))
        else:
            print("查询Oracle失败，未找到数据")
    except Exception as e:
        print("\n查询Oracle失败，完整错误信息:")
        print(f"异常类型: {type(e).__name__}")
        print(f"异常消息: {str(e)}")
        print("\n堆栈跟踪:")
        print(traceback.format_exc())

# 其他函数保持不变...

async def insert_qa_data():
    # 创建测试数据
    data = QAData(
        question="Oracle是什么?",
        answer="Oracle是一个开源的内存数据结构存储",
        qa_embedding=[0.4, 0.5, 0.6],
        references=[
            Reference(
                chunk_type=2,
                chunk_file_path="",
                file_ext=".pdf",
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
        request_time="2025-07-31 12:00:00",
        response_time="2025-07-31 12:00:01"
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

async def get_qa_data():
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

async def delete_session():
    if repo is None:
        print("创建会话存储库实例失败")
        return False
    result = await repo.delete_session(key)
    if result:
        print("删除Oracle成功")
    else:
        print("删除Oracle失败")

async def update_qa_feedback():
    try:
        if repo is None:
            print("创建会话存储库实例失败")
            return False
        print(f"正在写入Oracle，session_id: {key}")
        result = await repo.update_qa_feedback(key, 0, -3)
        if result:
            print("写入Oracle成功")
            await get_qa_data()
        else:
            print("写入Oracle失败")
    except Exception as e:
        print("\n写入Oracle失败，完整错误信息:")
        print(f"异常类型: {type(e).__name__}")
        print(f"异常消息: {str(e)}")
        print("\n堆栈跟踪:")
        print(traceback.format_exc())

async def get_last_qa_data():
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

async def update_last_qa_data_answer():
    try:
        if repo is None:
            print("创建会话存储库实例失败")
            return False
        print(f"正在写入Oracle，session_id: {key}")
        result = await repo.update_last_qa_data_answer(key, "您好！请问有什么可以帮助您的吗？")
        if result:
            print("写入Oracle成功")
            await get_last_qa_data()
        else:
            print("写入Oracle失败")
    except Exception as e:
        print("\n查询Oracle失败，完整错误信息:")
        print(f"异常类型: {type(e).__name__}")
        print(f"异常消息: {str(e)}")
        print("\n堆栈跟踪:")
        print(traceback.format_exc())

async def delete_by_agent():
    if repo is None:
        print("创建会话存储库实例失败")
        return False
    result = await repo.delete_by_agent_id(1)
    if result:
        print("删除Oracle成功")
    else:
        print("删除Oracle失败")

async def main():
    """主测试函数，确保连接正确关闭"""
    print("Starting Oracle test...")
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
    print("Oracle test finished.")

if __name__ == "__main__":
    asyncio.run(main())