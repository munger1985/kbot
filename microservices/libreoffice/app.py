import os
import subprocess
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import signal
import psutil

app = FastAPI()


# 配置从环境变量读取
PDF_SERVICE_PORT = int(os.getenv("PDF_SERVICE_PORT", "8000"))
LIBREOFFICE_PORT = int(os.getenv("LIBREOFFICE_PORT", "2002"))
UVICORN_WORKERS = int(os.getenv("UVICORN_WORKERS", "4"))

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

@app.on_event("startup")
async def startup():
    """服务启动时初始化"""
    # 启动LibreOffice
    app.state.libreoffice_process = start_libreoffice()
    
    # 预热连接池
    warmup_unoconv()
    
    # 打印系统信息
    logger.info(f"服务配置：Workers={UVICORN_WORKERS}, LibreOffice端口={LIBREOFFICE_PORT}")
    logger.info(f"系统资源：CPU={psutil.cpu_count()}核, 内存={psutil.virtual_memory().total/1024/1024:.1f}MB")

@app.on_event("shutdown")
async def shutdown():
    """服务关闭时清理"""
    if hasattr(app.state, 'libreoffice_process'):
        os.killpg(os.getpgid(app.state.libreoffice_process.pid), signal.SIGTERM)
        logger.info("已终止LibreOffice进程")

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
        app,
        host="0.0.0.0",
        port=PDF_SERVICE_PORT,
        workers=UVICORN_WORKERS
    )