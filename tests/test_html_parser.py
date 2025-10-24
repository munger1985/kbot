import sys
from pathlib import Path
import asyncio



project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from services.dataparse.html_parser import process_html
from services.dataparse.file_params import FileParams

if __name__ == "__main__":

    file = FileParams()
    file.file_id="1"
    file.app_id = 1
    file.kb_id = 30
    file.batch_id = 1
    file.file_path="/mnt/f/docs/Database Security and DBSAT Discover.html"
    file.file_ext = ".html"
    file.enable_summary = False
    file.kb_category = 1
    file.img2txt = 0
    file.tab_head = 0
    file.priority = 1
    file.parser = {}
    file.biz_metadata = {}
    file.img2txt_model = 30
    file.img_embed_model = None
    file.txt_embed_model = 21
    file.summary_model = 40
    file.security_level = 0
        

    asyncio.run(process_html(file))