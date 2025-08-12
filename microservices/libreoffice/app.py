import os
import sys
import subprocess
import time
import platform
import signal
import psutil
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from loguru import logger
# 添加项目根目录到 Python 路径，确保可以导入项目模块
current_file = os.path.abspath(__file__)
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from core.config import settings
from core.log.logger import setup_logging


# 配置从环境变量读取
PDF_SERVICE_PORT = int(os.getenv("PDF_SERVICE_PORT", "8000"))
LIBREOFFICE_PORT = int(os.getenv("LIBREOFFICE_PORT", "2002"))
UVICORN_WORKERS = int(os.getenv("UVICORN_WORKERS", "4"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager. //应用程序生命周期上下文管理器"""
    # 初始化日志
    setup_logging(service_name="Libre")
    
    # 启动事件
    start_time = time.time()
    logger.info(f"Initializing LibreOffice service at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...") 
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"Process ID: {os.getpid()}")
    
    # 启动LibreOffice服务
    try:
        logger.info(f"Starting LibreOffice service at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
        app.state.libreoffice_process = start_libreoffice()
        logger.info(f"LibreOffice版本: {subprocess.check_output(['libreoffice', '--version'], text=True).strip()}")
        warmup_unoconv()
    except Exception as e:
        logger.error(f"LibreOffice service initialization failed: {e}")
        # 在生产环境中，可能需要在这里退出应用程序
        current_env = os.getenv("KBOT_ENV")
        if current_env == "production":
            sys.exit(1)
    
    yield  # 服务运行期间
    
    # 关闭事件
    logger.info(f"Closing LibreOffice service at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
    shutdown_start = time.time()
    
    try:
        if hasattr(app.state, 'libreoffice_process'):
            os.killpg(os.getpgid(app.state.libreoffice_process.pid), signal.SIGTERM)
            logger.info("LibreOffice service is closed.")
    except Exception as e:
        logger.error(f"LibreOffice service shutdown failed: {e}")
    
    logger.info(f"LibreOffice service closed successfully, elapsed time: {time.time() - shutdown_start:.2f} seconds")
    logger.info(f"Total running time: {time.time() - start_time:.2f} seconds")

# 创建 FastAPI 应用
app = FastAPI(
    title="LibreOffice service",
    description="Provides LibreOffice services to convert word/ppt into pdf.",
    version=settings["app"]["version"],
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

class ConversionRequest(BaseModel):
    file_path: str

def start_libreoffice():
    """启动LibreOffice服务进程"""
    try:
        logger.info(f"启动LibreOffice服务 (Port: {LIBREOFFICE_PORT})")
        process = subprocess.Popen([
            'libreoffice',
            '--headless',
            '--invisible',
            '--nocrashreport',
            '--nodefault',
            '--nologo',
            '--nofirststartwizard',
            f'--accept=socket,host=0.0.0.0,port={LIBREOFFICE_PORT};urp;'
        ], preexec_fn=os.setsid)
        time.sleep(10)  # 等待服务初始化
        return process
    except Exception as e:
        logger.error(f"LibreOffice启动失败: {str(e)}")
        raise

def warmup_unoconv():
    """预热unoconv连接池"""
    logger.info("预热unoconv连接...")
    try:
        subprocess.run([
            'unoconv',
            '--listener',
            '--port', str(LIBREOFFICE_PORT),
            '--timeout', '10'
        ], check=True, timeout=15)
    except subprocess.TimeoutExpired:
        logger.warning("unoconv预热超时（正常现象）")
    except Exception as e:
        logger.error(f"预热失败: {str(e)}")
        raise



@app.post("/convert")
async def convert(request: ConversionRequest):
    """文件转换接口"""
    try:
        if not os.path.exists(request.file_path):
            raise HTTPException(status_code=404, detail="文件不存在")
        
        pdf_path = f"{os.path.splitext(request.file_path)[0]}.pdf"
        
        logger.info(f"开始转换: {os.path.basename(request.file_path)}")
        start_time = time.time()
        
        subprocess.run([
            'unoconv',
            '-f', 'pdf',
            '-e', 'SelectPdfVersion=1',
            '--port', str(LIBREOFFICE_PORT),
            '--timeout', str(os.getenv("LIBREOFFICE_TIMEOUT", "300")),
            request.file_path
        ], check=True, timeout=int(os.getenv("LIBREOFFICE_TIMEOUT", "300")) + 10)
        
        cost = time.time() - start_time
        logger.info(f"转换完成: {os.path.basename(pdf_path)} [耗时{cost:.2f}s]")
        
        return {
            "status": "success",
            "pdf_path": pdf_path,
            "time_used": f"{cost:.2f}s"
        }
    except subprocess.TimeoutExpired:
        logger.error("转换超时")
        raise HTTPException(status_code=504, detail="转换超时")
    except Exception as e:
        logger.error(f"转换失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    """监控指标端点"""
    return {
        "cpu_usage": f"{psutil.cpu_percent()}%",
        "memory_usage": f"{psutil.virtual_memory().percent}%",
        "libreoffice_port": LIBREOFFICE_PORT,
        "workers": UVICORN_WORKERS
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "microservices.libreoffice.app:app",
        host="0.0.0.0",
        port=PDF_SERVICE_PORT,
        workers=UVICORN_WORKERS
    )