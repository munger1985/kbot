import asyncio
import importlib
import sys
from pathlib import Path
from datetime import datetime
from typing import Any
from loguru import logger

# Add both project root and backend directory to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# 导入底层的统一图检索服务
from services.search.graph_search import GraphBaseSearch

async def debug_underlying_service():
    logger.info("🕵️ 开始应用层底层服务孤立调试...")
    
    # 实例化底层服务
    search_service = GraphBaseSearch()
    
    # 1. 第一轮实验：严格按照原本的入参跑（包含 kb_id = 101）
    logger.warning("🧪 实验 1: 使用默认 kb_id=101 尝试检索...")
    raw_bucket_1 = await search_service.search_by_graph(
        kb_id=101,  # 确认这里的 kb_id 数据库里是否存在
        vertex_names=["样品温度", "霍尔测量", "磁场强度"],
        search_top_k=5,
        weight=1.5,
        security=9,
        max_depth=2
    )
    logger.info(f"实验 1 结果: {raw_bucket_1}")

    # 2. 第二轮实验：改变输入，直接将 SQL 中已经证实的、存在直接扩散边的词传进去
    logger.warning("🧪 实验 2: 使用图谱强关联枢纽词 '霍尔因子' 尝试破冰匹配...")
    raw_bucket_2 = await search_service.search_by_graph(
        kb_id=101, 
        vertex_names=["霍尔因子", "n型硅"], 
        search_top_k=5,
        weight=1.5,
        security=9,
        max_depth=2
    )
    logger.info(f"实验 2 结果 (是否成功捞出数据): {raw_bucket_2}")

if __name__ == "__main__":
    asyncio.run(debug_underlying_service())