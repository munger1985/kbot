"""解析参数定义模型。"""

import json
from pydantic import BaseModel, Field, field_validator

class DocParserParams(BaseModel):
    """文档解析任务参数配置。"""
    chunk_size: int = Field(800, ge=64, description="分块最大字符数（增大以保语义完整）")
    min_chunk_len: int = Field(100, ge=50, description="最小分块字符长度")
    generate_picture_images: bool = Field(True, description="是否提取文档图片")
    image_scale: float = Field(2.0, gt=0.0, description="图片渲染缩放比例")
    image_dir: str | None = Field(None, description="抽取图片保存路径")
    do_ocr: bool = Field(True, description="是否启用 OCR 识别扫描 PDF 中的文字")
    ocr_engine: str | None = Field("easyocr", description="指定 OCR 引擎 (easyocr, tesseract)")
    ocr_model: str | None = Field(None, description="指定 AI OCR 模型名称 (如 DeepSeek OCR)，优先级高于内置 OCR")
    vlm_model: str | None = Field(None, description="指定 VLM 模型名称（非空时启用 VLM 增强）")
    llm_model: str = Field(..., description="指定 LLM 模型名称（全局摘要/元数据提取）")
    img2txt_prompt: str = Field(..., description="自定义 VLM 提取图片提示词")
    enable_layout_clustering: bool = Field(True, description="启用多栏检测")
    enable_page_span_stitch: bool = Field(True, description="启用跨页段落缝合")
    enable_doc_metadata: bool = Field(True, description="提取文档元数据到 PG")
    engine_mode: str = Field("auto", description="PDF 解析方式: vlm(视觉解析) / auto(文本提取)。Word/PPT/Excel 始终用 Docling 不受此影响")
    enable_chunk_reflection: bool = Field(False, description="启用 LLM 后反思重组短 chunk（仅在 auto/precision 模式下生效）")
    visual_model: str = Field("", description="视觉嵌入模型名称（选填，用于生成图片向量并入库，支持以图搜图）")
    kb_id: str = Field("", description="知识库ID（由调用方自动填充）")
    # 移除: extract_graph, overlap

    @property
    def effective_do_ocr(self) -> bool:
        """自动推导：仅当没有配置 AI OCR 模型时，才启用 docling 内置 OCR 兜底。

        当配置了 DeepSeek OCR 等 AI 模型时，文字提取由 AI 模型在 post-processing 阶段完成，
        docling 内置 OCR 不再需要运行（节省处理时间）。
        """
        return self.ocr_model is None

class FileParams(BaseModel):
    file_id: str = Field(..., description="文件ID")
    kb_id: str = Field(..., description="知识库ID")
    file_path: str = Field(..., description="文件路径")
    file_ext: str = Field(..., description="文件扩展名")
    priority: int = Field(0, description="处理优先级")
    security_level: int = Field(0, description="文件安全等级")
    parser_params: DocParserParams = Field(..., description="解析器配置")
    biz_metadata: dict = Field({}, description="业务元数据")
    txt_embed_model: str | None = Field(None, description="文本嵌入模型ID")

    @field_validator('biz_metadata', mode='before')
    @classmethod
    def parse_biz_metadata(cls, v):
        """处理 asyncpg 将 JSON 列返回为字符串的情况"""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return {}
        return v if v is not None else {}

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
    # 新增：层级感知字段
    hierarchy_path: list[str] = []
    hierarchy_depth: int = 0
    heading_level: int = 0
    parent_chunk_id: str | None = None
    section_id: str | None = None

    @classmethod
    def create(cls, content: str, summary: str, header: str, search_helper: str,
               chunk_num: int, chunk_type: str, metadata: ChunkMetadata,
               hierarchy_path: list[str] | None = None,
               hierarchy_depth: int = 0,
               heading_level: int = 0,
               parent_chunk_id: str | None = None,
               section_id: str | None = None,
               ):
        """统一构造工厂"""
        return cls(
            content=content.strip(),
            doc_summary=summary,
            header=header,
            search_helper=search_helper,
            chunk_type=chunk_type,
            chunk_num=chunk_num,
            metadata=metadata,
            hierarchy_path=hierarchy_path or [],
            hierarchy_depth=hierarchy_depth,
            heading_level=heading_level,
            parent_chunk_id=parent_chunk_id,
            section_id=section_id,
        )