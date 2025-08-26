import sys
from pathlib import Path
import asyncio

# Add both project root and backend directory to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from services.search.chinese_preprocessor import preprocess_cn_query

async def test_preprocess_cn_query():
    # 示例查询
    test_queries = [
        "郑和下西洋是什么时候?？"
    ]

    for query in test_queries:
        # 用于语义检索（返回字符串）
        processed = await preprocess_cn_query(query)
        print(f"原始: {query}")
        print(f"语义检索用: {processed["semantic"]}") # type: ignore
        print(f"全文检索用: {processed["fulltext"]}") # type: ignore
        print("=" * 50)

if __name__ == "__main__":
    asyncio.run(test_preprocess_cn_query())