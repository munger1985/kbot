import io
import base64
import asyncio
from PIL import Image
from decimal import Decimal
from json import JSONEncoder
    

class DecimalEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            # 对于整数部分保持整数类型
            if obj == obj.to_integral_value():
                return int(obj)
            return float(obj)
        return super().default(obj)
    

class ImageEncoder:

    @staticmethod
    async def encode(image: str | Image.Image) -> str:
        """将图像转换为 Base64 编码。

        针对内存中的 PIL 对象或本地路径进行统一编码，并校验文件大小。

        Args:
            image: 图像文件路径或 PIL 图像对象。

        Returns:
            str: UTF-8 编码的 Base64 字符串。

        Raises:
            ValueError: 图像大小超过 20MB 时抛出。
        """
        loop = asyncio.get_running_loop()

        def _process():
            # 分支 1：如果是文件路径 (str)
            if isinstance(image, str):
                with open(image, "rb") as f:
                    data = f.read()
            # 分支 2：如果是 PIL 图像对象
            else:
                buf = io.BytesIO()
                # 先转 RGB 避免 RGBA 转 JPEG 报错，再进行保存
                rgb_image = image.convert("RGB")
                rgb_image.save(buf, format="JPEG", quality=85)
                data = buf.getvalue()
            return data

        # 在执行器中运行同步阻塞操作
        img_data = await loop.run_in_executor(None, _process)

        if len(img_data) > 20 * 1024 * 1024:
            raise ValueError("图像大小超过 20MB 限制")

        return base64.b64encode(img_data).decode('utf-8')