from typing import Optional

class FileParams:
    def __init__(self):
        self.file_id: int = 0
        self.app_id: int = 0
        self.kb_id: int = 0
        self.batch_id: Optional[int] = None
        self.file_path: str = ""
        self.file_ext: Optional[str] = None
        self.summary: int = 0
        self.kb_category: int = 0
        self.img2txt: int = 0
        self.tab_head: int = 0
        self.priority: int = 0
        self.parser: dict = {}
        self.img2txt_model: Optional[int] = None
        self.img_embed_model: Optional[int] = None
        self.txt_embed_model: Optional[int] = None