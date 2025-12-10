import sys
from pathlib import Path
import asyncio

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from services.dataparse.markdown_parser_bs4 import process_markdown
from services.dataparse.file_params import FileParams

if __name__ == "__main__":
    file = FileParams()
    file.file_id = "8aedef95-5e23-4517-91d0-c93476ae1c68"
    file.app_id = 112
    file.kb_id = 544
    file.img2txt_model = 68
    file.txt_embed_model = 65
    file.summary_model = 67
    file.enable_summary = False
    file.img2txt = 0
    file.batch_id = 643
    file.file_path = "/home/ubuntu/kbot_data/files/221/544/source/batch1/hub_env.md"
    file.file_ext = ".md"
    file.kb_category = 1
    file.tab_head = 0
    file.priority = 1
    file.parser = {}
    file.biz_metadata = {}
    file.img_embed_model = None
    file.security_level = 0

    asyncio.run(process_markdown(file))