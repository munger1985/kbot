from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.services.office_service import OfficeService
from app.config import Config
import os
import tempfile
from typing import Optional

app = FastAPI()
config = Config()
office = OfficeService()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/convert/pdf")
async def convert_to_pdf(file: UploadFile = File(...)):
    try:
        # 验证文件大小
        file.file.seek(0, 2)
        file_size = file.file.tell()
        if file_size > config.MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="文件过大")
        file.file.seek(0)
        
        # 保存临时文件
        temp_file = tempfile.NamedTemporaryFile(delete=False, dir=config.DOCUMENTS_DIR)
        temp_file.write(file.file.read())
        temp_file.close()
        
        # 转换PDF
        pdf_path = office.convert_to_pdf(temp_file.name)
        
        # 返回PDF文件
        return JSONResponse({
            "status": "success",
            "pdf_path": pdf_path,
            "file_size": os.path.getsize(pdf_path)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)

@app.post("/extract/page")
async def extract_page(
    file: UploadFile = File(...),
    page_number: int = 1,
    dpi: Optional[int] = None
):
    try:
        # 验证文件大小
        file.file.seek(0, 2)
        file_size = file.file.tell()
        if file_size > config.MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="文件过大")
        file.file.seek(0)
        
        # 保存临时文件
        temp_file = tempfile.NamedTemporaryFile(delete=False, dir=config.DOCUMENTS_DIR)
        temp_file.write(file.file.read())
        temp_file.close()
        
        # 提取页面
        result = office.extract_page_as_image(temp_file.name, page_number, dpi)
        
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=config.API_PORT)