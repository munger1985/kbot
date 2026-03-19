"""Definition models for parsing parameters."""

from pydantic import BaseModel, Field

class DocParserParams(BaseModel):
    """Configuration parameters for document parsing tasks.
    """
    chunk_size: int = Field(512, ge=64, description="Maximum token count per chunk")
    overlap: int = Field(50, ge=0, description="Token overlap between adjacent chunks")
    min_chunk_len: int = Field(10, ge=1, description="Minimum character length of chunks")
    generate_picture_images: bool = Field(True, description="Whether to extract document images")
    image_scale: float = Field(2.0, gt=0.0, description="Image rendering scale ratio")
    image_dir: str | None = Field(None, description="Storage path for extracted images")
    do_ocr: bool = Field(True, description="Whether to enable OCR for text recognition in scanned PDFs")
    ocr_engine: str | None = Field("easyocr", description="Specify OCR engine (easyocr, tesseract)")
    use_vlm: bool = Field(True, description="Whether to enable VLM semantic enhancement")
    vlm_model: str | None = Field(None, description="Specify VLM model name")
    vlm_prompt: str | None = Field(None, description="Custom VLM prompt text")


class FileParams(BaseModel):
    """Parameters for file processing with business context.
    """
    file_id: str = Field(..., description="File ID")
    kb_id: int = Field(..., description="Knowledge base ID")
    file_path: str = Field(..., description="File path")
    file_ext: str = Field(..., description="File extension")
    priority: int = Field(0, description="Processing priority")
    security_level: int = Field(0, description="File security level")
    parser_params: DocParserParams = Field(..., description="Parser configuration")
    biz_metadata: dict = Field({}, description="Business metadata")
    txt_embed_model: str | None = Field(None, description="Text embedding model ID")