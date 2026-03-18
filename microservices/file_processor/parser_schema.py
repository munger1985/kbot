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
    use_vlm: bool = Field(True, description="是否启用 VLM 语义增强")
    vlm_model: str | None = Field(None, description="指定 VLM 模型名称")
    vlm_prompt: str | None = Field(None, description="自定义 VLM 提示词")


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