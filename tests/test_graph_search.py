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

async def final_debug_tool():
    logger.info("⚙️ 启动终极全链路对齐测试...")
    search_service = GraphBaseSearch()
    
    # =================================================================
    # 实验 A：验证“大小写/别名”导致的崩溃
    # =================================================================
    logger.warning("🧪 1. 尝试使用 SQL 中已被证实的枢纽节点组合进行验证...")
    try:
        # 如果你已经修改了 graph_repo.py 里的 SOURCE_ID 提取大小写，这里应该能直接过
        res_a = await search_service.search_by_graph(
            kb_id=101, 
            vertex_names=["霍尔因子", "n型硅"], 
            search_top_k=5,
            weight=1.5,
            security=9,
            max_depth=2
        )
        logger.success(f"🎉 实验 A 破冰成功！召回数据结构: {res_a}")
    except Exception as e:
        logger.error(f"❌ 实验 A 仍然触发解析崩溃: {e}")
        logger.info("💡 请进入 dao/repositories/graph_repo.py 约 268 行，"
                    "检查获取 row 字段时是否使用了 row['SOURCE_ID']，将其统一改为小写 row['source_id'] 或根据驱动要求转换。")

    # =================================================================
    # 实验 B：攻克实验 1 返回 0 的多度图下游走断层
    # =================================================================
    logger.warning("🧪 2. 验证多度图下游走（最大深度=2）...")
    # 根据你之前成功的 SQL，['样品温度', '霍尔测量', '磁场强度'] 需要外扩一度才能撞击到 '霍尔因子'
    # 如果把中介词一起输入，看是否能稳定拿到结果
    extended_entities = ["样品温度", "霍尔测量", "磁场强度", "霍尔因子"]
    logger.info(f"注入带中介枢纽的外扩实体集合: {extended_entities}")
    
    try:
        res_b = await search_service.search_by_graph(
            kb_id=101, 
            vertex_names=extended_entities, 
            search_top_k=5,
            weight=1.5,
            security=9,
            max_depth=2
        )
        logger.success(f"📊 实验 B 拓扑放大测试完成，召回结果数: {len(res_b.get('graph_result', []))}")
    except Exception as e:
        logger.error(f"❌ 实验 B 游走异常: {e}")


if __name__ == "__main__":
    asyncio.run(final_debug_tool())