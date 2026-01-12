"""解析参数定义模型。"""

from pydantic import BaseModel, Field
from fastapi import Form

class ParserParams(BaseModel):
    """文档解析任务参数配置。"""

    # --- 基础配置 ---
    file_path: str = Field(..., description="本地或临时文件路径")
    output_format: str = Field("markdown", description="格式: markdown, html, json, doctags, chunks")

    # --- 切分参数 ---
    chunk_size: int = Field(512, ge=64, description="分块最大 Token 数")
    overlap: int = Field(50, ge=0, description="分块重叠 Token 数")
    min_chunk_len: int = Field(10, ge=1, description="最小分块字符长度")

    # --- 转换配置 ---
    do_ocr: bool = Field(True, description="是否启用 OCR")
    generate_picture_images: bool = Field(True, description="是否提取文档图片")
    images_scale: float = Field(2.0, gt=0.0, description="图片渲染缩放比例")

    # --- VLM 增强配置 ---
    use_vlm: bool = Field(True, description="是否启用 VLM 语义增强")
    vlm_model: int | None = Field(None, description="指定 VLM 模型ID")
    vlm_prompt: str | None = Field(None, description="自定义 VLM 提示词")

    def to_dict(self) -> dict:
        """模型转字典。"""
        return self.model_dump()
    
    @classmethod
    def as_form(
        cls,
        output_format: str = Form("markdown"),
        chunk_size: int = Form(512),
        overlap: int = Form(50),
        min_chunk_len: int = Form(10),
        do_ocr: bool = Form(True),
        generate_picture_images: bool = Form(True),
        images_scale: float = Form(2.0),
        use_vlm: bool = Form(True),
        vlm_model: int | None = Form(None),
        vlm_prompt: str | None = Form(None),
    ):
        return cls(
            file_path="",  # 初始为空，由 Endpoint 填充
            output_format=output_format,
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk_len=min_chunk_len,
            do_ocr=do_ocr,
            generate_picture_images=generate_picture_images,
            images_scale=images_scale,
            use_vlm=use_vlm,
            vlm_model=vlm_model,
            vlm_prompt=vlm_prompt
        )