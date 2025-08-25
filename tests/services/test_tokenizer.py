import sys
from pathlib import Path

# Add both project root and backend directory to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
from services.search.chinese_preprocessor import preprocess_cn_query

# 示例查询
test_queries = [
    "郑和下西洋是什么时候?？"
]

for query in test_queries:
    # 用于语义检索（返回字符串）
    processed_str = preprocess_cn_query(query, return_string=True)
    print(f"原始: {query}")
    print(f"语义检索用: {processed_str}")
    
    # 用于全文检索（返回词元列表）
    processed_tokens = preprocess_cn_query(query, return_string=False)
    print(f"全文检索用: {processed_tokens}")
    print("-" * 50)