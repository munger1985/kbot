import asyncio
import os
import sys
import aiohttp
from pathlib import Path
from dotenv import load_dotenv


# Add both project root and backend directory to Python path
project_root = Path(__file__).resolve().parent.parent.parent.parent
backend_dir = project_root / "backend"
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(project_root))

# Use absolute imports from project root
from dao.repositories.kbot_biz_txt_embedding_repo import KbotBizTxtEmbeddingRepository
from utils.oracle_vec_handler import OracleVecHandler
from services.chat.agent_chat import Agent

async def main():
    """End-to-end test with real embedding service and database"""
    # Load environment variables
    load_dotenv()
    
    # 1. Prepare test text
    test_text = "文艺复兴是什么？"
    
    agent = Agent()
    r = await agent.chat(1, test_text)
    for res in r:
        print(res.embed_id)
        chunk = await lob_to_string(res.chunk_doc)
        print(chunk)
        print(res.similarity)


async def lob_to_string(async_lob):
    """
    将 AsyncLOB 对象转换为字符串
    :param async_lob: oracledb.AsyncLOB 对象
    :return: 字符串内容
    """
    content = await async_lob.read()
    if isinstance(content, bytes):
        return content.decode('utf-8')  # 假设使用UTF-8编码
    return content

if __name__ == '__main__':
    print("Starting embedding similarity test...")
    asyncio.run(main())