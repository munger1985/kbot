from fastapi import FastAPI, UploadFile, Form, HTTPException, Response
from pathlib import Path
import shutil
import os
from io import BytesIO
from tempfile import mkdtemp
from app.converter import OfficeToPDF
from PyPDF2 import PdfReader, PdfWriter

app = FastAPI()
converter = OfficeToPDF()

@app.post("/convert")
async def convert_office(
    file: UploadFile,
    page: int = Form(None)
):
    if file.filename is None:
        raise HTTPException(status_code=400, detail="Invalid file")
    
    # 创建临时文件夹并临时保存输入的文件
    temp_dir = mkdtemp()

    try:
        
        # 第一步：将上传文件临时保存
        file_name = os.path.basename(file.filename)
        temp_path = os.path.join(temp_dir, file_name)
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # 第二步：将 Office 文件转成 PDF
        pdf_filename = Path(file.filename).stem + ".pdf"
        pdf_path = os.path.join(temp_dir, pdf_filename)

        await OfficeToPDF().convert_to_pdf(input_path=temp_path, output_path=pdf_path)

        
        if page is None:
            # 返回整个PDF的二进制流
            with open(pdf_path, 'rb') as f:
                file_content = f.read()
            return Response(content=file_content, media_type='application/pdf') 
        
        else:
            # 提取指定页并直接返回 PDF 二进制流
            
            
            reader = PdfReader(pdf_path)
            if page < 1 or page > len(reader.pages):
                raise HTTPException(status_code=400, detail="Page number out of range")
            
            writer = PdfWriter()
            writer.add_page(reader.pages[page - 1])
                   
            pdf_byte_arr = BytesIO()
            writer.write(pdf_byte_arr)
            return Response(content=pdf_byte_arr.getvalue(), media_type='application/pdf')

            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 清理临时文件
        shutil.rmtree(temp_dir)
        
        


# if __name__ == "__main__":
#     import uvicorn
    
#     service_host = "0.0.0.0"
#     service_port = 20303
    
#     uvicorn.run(app, host=service_host, port=service_port)