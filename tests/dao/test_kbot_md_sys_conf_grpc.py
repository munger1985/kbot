import asyncio
import grpc
import os
import sys
from grpc import aio
# 添加项目根目录到 Python 路径，确保可以导入项目模块
current_file = os.path.abspath(__file__)
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
print("backend_dir:", backend_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
from dao.metadata_service.kbot_md_sys_conf_pb2 import KbotMdSysConf, Empty
from dao.metadata_service.kbot_md_sys_conf_pb2_grpc import KbotMdSysConfServiceStub

async def query_database():
    """调用 gRPC 查询数据库"""
    async with aio.insecure_channel('localhost:50051') as channel:
        stub = KbotMdSysConfServiceStub(channel)
        response = await stub.GetAll(Empty())
        print("Response:", response)

if __name__ == "__main__":
    asyncio.run(query_database())
    """创建 gRPC 通道"""
    
