from pydantic import BaseModel, Field


class ParserParams(BaseModel):
    """解析参数模型"""
    chunk_size: int = Field(..., description="分块大小")
    overlap: int = Field(..., description="分块重叠大小")
    min_chunk_len: int = Field(..., description="最小分块长度")
    generate_picture_images: bool = Field(description="是否生成图片描述")
    do_ocr: bool = Field(description="是否进行OCR识别")
    ocr_engine: str | None = Field(None, description="OCR引擎名称")
    images_scale: float = Field(..., description="图片缩放比例")
    use_vlm: bool = Field(default=False, description="是否使用VLM生成图片描述")
    vlm_model: int | None = Field(None, description="VLMParser模型ID")
    vlm_prompt: str | None = Field(None, description="VLMParser提示词名称")

    def to_dict(self) -> dict:
        """转换为字典"""
        return self.model_dump()