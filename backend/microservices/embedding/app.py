import os
import sys
import uvicorn
import signal
import subprocess
import atexit
from typing import List, Optional
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
import numpy as np
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# 添加项目根目录到 Python 路径，确保可以导入项目模块
# 获取当前文件的绝对路径
current_file = os.path.abspath(__file__)
# 获取 backend 目录的路径（假设当前文件在 backend/microservices/embedding/ 目录下）
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
# 将 backend 目录添加到 Python 路径
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# 现在可以导入项目模块了
from core.config import settings
from services.embedding import EmbeddingService, embedding_service
from dao.repositories.kbot_md_models_repo import KbotMdModelsRepository

# 确保日志目录存在
log_dir = settings["logger"]["dir"]
os.makedirs(log_dir, exist_ok=True)

# 配置日志 - 使用 loguru，覆盖日志文件路径
logger.add(
    os.path.join(log_dir, "embedding_service.log"),
    rotation=settings["logger"]["rotation"],
    retention=settings["logger"]["retention"],
    level=settings["logger"]["level"]
)

# 定义 lifespan 上下文管理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动事件
    await embedding_service.initialize()
    logger.info("嵌入服务已初始化")
    
    yield  # 服务运行期间
    
    # 关闭事件
    await embedding_service.shutdown()
    logger.info("嵌入服务已关闭")

# 创建 FastAPI 应用
app = FastAPI(
    title="嵌入微服务",
    description="提供文本嵌入服务，将文本转换为向量表示",
    version="1.0.0",
    lifespan=lifespan,
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该限制为特定的源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定义请求模型
class EmbeddingRequest(BaseModel):
    model_id: int = Field(..., description="嵌入模型的ID")
    texts: List[str] = Field(..., description="要嵌入的文本列表")
    batch_size: Optional[int] = Field(32, description="批处理大小")

# 定义响应模型
class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]] = Field(..., description="嵌入向量列表")
    model_id: int = Field(..., description="使用的嵌入模型ID")
    dimensions: int = Field(..., description="嵌入向量的维度")

# 依赖项：获取嵌入服务实例
def get_embed_service():
    return embedding_service

# 启动和关闭事件已移至 lifespan 上下文管理器

@app.post("/embed", response_model=EmbeddingResponse)
async def embed_texts(
    request: EmbeddingRequest,
    embed_service: EmbeddingService = Depends(get_embed_service)
):
    """
    将文本列表转换为嵌入向量
    
    - **model_id**: 要使用的嵌入模型的ID
    - **texts**: 要嵌入的文本列表
    - **batch_size**: 批处理大小（可选，默认为32）
    
    返回:
    - **embeddings**: 嵌入向量列表
    - **model_id**: 使用的嵌入模型ID
    - **dimensions**: 嵌入向量的维度
    """
    try:
        logger.info(f"接收到嵌入请求: 模型={request.model_id}, 文本数量={len(request.texts)}")
        
        # 使用嵌入服务将文本转换为向量
        embeddings = await embed_service.embed_texts(
            model_id=request.model_id,
            texts=request.texts,
            batch_size=request.batch_size
        )
        
        # 将numpy数组转换为Python列表
        embeddings_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
        
        # 获取嵌入向量的维度
        dimensions = len(embeddings_list[0]) if embeddings_list and len(embeddings_list) > 0 else 0
        
        logger.info(f"嵌入完成: 向量数量={len(embeddings_list)}, 维度={dimensions}")
        
        return {
            "embeddings": embeddings_list,
            "model_id": request.model_id,
            "dimensions": dimensions
        }
    except Exception as e:
        logger.error(f"嵌入过程中出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"嵌入过程中出错: {str(e)}")

@app.get("/embed/models")
async def list_models(
    embed_service: EmbeddingService = Depends(get_embed_service)
):
    """
    获取可用的嵌入模型列表
    
    返回:
    - 可用模型ID的列表
    """
    try:
        # 从数据库获取模型列表
        
        md_repo = KbotMdModelsRepository()
        models = await md_repo.get_all_embedding_models()
        result = []
        for model in models:
            result.append({"model_id": model.model_id, "model_name": model.model_name})
        return result
    except Exception as e:
        logger.error(f"获取模型列表时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取模型列表时出错: {str(e)}")

@app.get("/embed/health")
async def health_check():
    """
    健康检查端点
    
    返回:
    - 服务状态信息
    """
    return {"status": "healthy", "service": "embedding-service"}

# 全局变量，用于存储微服务进程
embedding_service_process = None

def start_embedding_service():
    """启动嵌入微服务作为独立进程"""
    logger.info("启动嵌入微服务作为独立进程")
    embedding_service_path = os.path.abspath(__file__)
    
    # 启动嵌入微服务，使用不同的端口（8001）并设置为独立模式
    process = subprocess.Popen(
        [sys.executable, embedding_service_path],
        env={**os.environ, "PORT": "8001", "EMBEDDING_SERVICE_STANDALONE": "1"}
    )
    return process

def shutdown_embedding_service():
    """关闭嵌入微服务进程"""
    global embedding_service_process
    if embedding_service_process:
        logger.info("关闭嵌入微服务进程")
        try:
            embedding_service_process.terminate()
            embedding_service_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("嵌入微服务进程未能正常终止，强制关闭...")
            embedding_service_process.kill()
        embedding_service_process = None

def signal_handler(sig, frame):
    """处理终止信号"""
    logger.info(f"接收到信号 {sig}，正在关闭...")
    shutdown_embedding_service()
    sys.exit(0)

# 注册退出处理程序，确保在应用程序退出时关闭微服务
atexit.register(shutdown_embedding_service)

if __name__ == "__main__":
    # 从环境变量获取主机和端口，如果没有设置，则使用默认值
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8001))
    
    # 如果是作为独立进程启动，则注册信号处理器
    if os.environ.get("EMBEDDING_SERVICE_STANDALONE") == "1":
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info(f"启动嵌入微服务，监听 {host}:{port}")
    uvicorn.run(app, host=host, port=port)