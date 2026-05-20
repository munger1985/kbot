"""解析参数定义模型。"""

from pydantic import BaseModel, Field

class DocParserParams(BaseModel):
    """文档解析任务参数配置。"""
    chunk_size: int = Field(512, ge=64, description="分块最大 Token 数")
    overlap: int = Field(50, ge=0, description="分块重叠 Token 数")
    min_chunk_len: int = Field(10, ge=1, description="最小分块字符长度")
    generate_picture_images: bool = Field(True, description="是否提取文档图片")
    image_scale: float = Field(2.0, gt=0.0, description="图片渲染缩放比例")
    image_dir: str | None = Field(None, description="抽取图片保存路径")
    do_ocr: bool = Field(True, description="是否启用 OCR 识别扫描 PDF 中的文字")
    ocr_engine: str | None = Field("easyocr", description="指定 OCR 引擎 (easyocr, tesseract)")
    use_vlm: bool = Field(True, description="是否使用全量 VLM 解析")
    vlm_model: str | None = Field(None, description="指定 VLM 模型名称")
    llm_model: str = Field(..., description="指定 LLM 模型名称")
    img2txt_prompt: str = Field(..., description="自定义 VLM 提取图片提示词")
    extract_graph: bool = Field(False, description="是否提取图实体")

class FileParams(BaseModel):
    file_id: str = Field(..., description="文件ID")
    kb_id: int = Field(..., description="知识库ID")
    file_path: str = Field(..., description="文件路径")
    file_ext: str = Field(..., description="文件扩展名")
    priority: int = Field(0, description="处理优先级")
    security_level: int = Field(0, description="文件安全等级")
    parser_params: DocParserParams = Field(..., description="解析器配置")
    biz_metadata: dict = Field({}, description="业务元数据")
    txt_embed_model: str | None = Field(None, description="文本嵌入模型ID")

class ChunkMetadata(BaseModel):
    """增强型元数据"""
    page_num: int
    image_name: str | None = None
    bbox: list[float] | None = None # [x1, y1, x2, y2] 归一化坐标
    is_sub_table: bool = False
    
class ChunkResult(BaseModel):
    """全系统统一的 Chunk 结构"""
    content: str
    doc_summary: str
    header: str
    search_helper: str
    chunk_type: str  # text, table, picture, slide
    chunk_num: int
    metadata: ChunkMetadata

    @classmethod
    def create(cls, content: str, summary: str, header: str, search_helper: str,
               chunk_num: int, chunk_type: str, metadata: ChunkMetadata):
        """统一构造工厂"""
        return cls(
            content=content.strip(),
            doc_summary=summary,
            header=header,
            search_helper=search_helper,
            chunk_type=chunk_type,
            chunk_num=chunk_num,
            metadata=metadata
        )