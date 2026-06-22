from pydantic import BaseModel, Field


class ParserParams(BaseModel):
    """Parser parameters model.
    
    This model defines the configuration parameters for document parsing, including
    text chunking settings, image processing options, and OCR/VLM configurations.
    """
    chunk_size: int = Field(..., description="Chunk size (maximum length of each text chunk)")
    overlap: int = Field(..., description="Chunk overlap size (number of overlapping characters between adjacent chunks)")
    min_chunk_len: int = Field(..., description="Minimum chunk length (minimum valid length for a text chunk)")
    generate_picture_images: bool = Field(description="Whether to generate image descriptions (for picture content)")
    do_ocr: bool = Field(description="Whether to perform OCR recognition (extract text from images)")
    ocr_engine: str | None = Field(None, description="OCR engine name (specific OCR engine to use, optional)")
    ocr_model: str | None = Field(None, description="AI OCR model name (e.g. DeepSeek OCR), takes priority over built-in OCR")
    images_scale: float = Field(..., description="Image scaling ratio (scale factor for processing images)")
    use_vlm: bool = Field(default=False, description="Whether to use VLM for generating image descriptions")
    vlm_model: int | None = Field(None, description="VLMParser model ID (unique identifier for VLM model, optional)")
    vlm_prompt: str | None = Field(None, description="VLMParser prompt (prompt text for VLM model, optional)")

    def to_dict(self) -> dict:
        """Convert model instance to dictionary.
        
        Returns:
            dict: Dictionary containing all model fields and their values.
        """
        return self.model_dump()