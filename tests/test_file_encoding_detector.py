import os
import sys
import asyncio
# 添加项目根目录到 Python 路径，确保可以导入项目模块
current_file = os.path.abspath(__file__)
backend_dir = os.path.dirname(os.path.dirname(current_file))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
from services.dataparse.common import detect_file_encoding

if __name__ == "__main__":
    # 测试文件路径
    test_file = "/mnt/f/docs/优化查询DBA_HIST_ACTIVE_SESS_HISTORY慢的问题.txt"
    
    # 检测编码
    encoding = detect_file_encoding(test_file)
    print(f"检测到的编码: {encoding}")