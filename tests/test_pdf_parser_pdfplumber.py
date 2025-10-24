import sys
from pathlib import Path
import asyncio

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from services.dataparse.pdf_parser_pdfplumber import process_pdf
from services.dataparse.file_params import FileParams

if __name__ == "__main__":
    file = FileParams()
    file.file_id = "9515695f-a415-4010-95eb-8ee2e3e93e26"
    file.app_id = 1
    file.kb_id = 382
    file.img2txt_model = 68
    file.txt_embed_model = 65
    file.summary_model = 67
    file.enable_summary = False
    file.img2txt = 0
    file.batch_id = 1
    file.file_path = "/home/ubuntu/kbot_data/files/241/382/source/33/dapdf.pdf"
    file.file_ext = ".pdf"
    file.kb_category = 1
    file.tab_head = 0
    file.priority = 1
    file.parser = {}
    file.biz_metadata = {}
    file.img_embed_model = None
    file.security_level = 0

    asyncio.run(process_pdf(file))