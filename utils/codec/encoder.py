import io
import base64
import asyncio
from PIL import Image
from decimal import Decimal
from json import JSONEncoder
    

class DecimalEncoder(JSONEncoder):
    """JSON encoder for handling Decimal type values.
    
    Converts Decimal values to appropriate Python numeric types (int/float)
    for JSON serialization, preserving integer values as integers instead of floats.
    """
    def default(self, obj):
        if isinstance(obj, Decimal):
            # Preserve integer values as int type instead of float
            if obj == obj.to_integral_value():
                return int(obj)
            return float(obj)
        return super().default(obj)
    

class ImageEncoder:
    """Asynchronous image encoder for converting images to Base64 format.
    
    Provides static methods to encode images from file paths or PIL Image objects
    to Base64 strings, with size validation and format handling.
    """

    @staticmethod
    async def encode(image: str | Image.Image) -> str:
        """Convert image to Base64 encoded string asynchronously.

        Unified encoding for in-memory PIL Image objects or local file paths,
        with file size validation to ensure compliance with size limits.

        Args:
            image: Path to image file (string) or PIL Image object in memory.

        Returns:
            str: UTF-8 encoded Base64 string representation of the image.

        Raises:
            ValueError: Raised when image size exceeds 20MB limit.
        """
        loop = asyncio.get_running_loop()

        def _process():
            # Branch 1: If input is file path (string)
            if isinstance(image, str):
                with open(image, "rb") as f:
                    data = f.read()
            # Branch 2: If input is PIL Image object
            else:
                buf = io.BytesIO()
                # Convert to RGB first to avoid RGBA -> JPEG conversion errors, then save
                rgb_image = image.convert("RGB")
                rgb_image.save(buf, format="JPEG", quality=85)
                data = buf.getvalue()
            return data

        # Run synchronous blocking operations in executor to avoid event loop blocking
        img_data = await loop.run_in_executor(None, _process)

        if len(img_data) > 20 * 1024 * 1024:
            raise ValueError("Image size exceeds 20MB limit")

        return base64.b64encode(img_data).decode('utf-8')