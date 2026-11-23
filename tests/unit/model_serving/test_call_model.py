import asyncio
import os
from pathlib import Path
from uuid import UUID
from PIL import Image, ImageDraw, ImageFont
import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("KBOT_RUN_MODEL_SMOKE") != "1",
    reason="需要已启动并配置真实模型的四类推理服务",
)

# 加载 .env 文件（必须在导入任何使用配置的模块之前）
from dotenv import load_dotenv
env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(env_path)

from platform_clients.model import AIModelClient
from model_serving.config import get_model_serving_settings

settings = get_model_serving_settings()
model_client = AIModelClient(
    caller_service="model-client-integration-test",
    embedding_config=settings.embedding,
    llm_config=settings.llm,
    vlm_config=settings.vlm,
    visual_config=settings.visual,
)


def configured_model_id(kind: str) -> str:
    name = f"KBOT_SMOKE_{kind}_MODEL_ID"
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"请在 {name} 中设置待测试模型的 UUID")
    return str(UUID(value))


def create_basic_shapes_image() -> Image.Image: # 类型注解可以更精确为 Image.Image
    """
    创建一张包含基本形状（圆形、正方形、三角形）和对应标签的图片。
    该函数不将图片保存到磁盘，而是直接返回一个PIL Image对象。

    Returns:
        Image.Image: 包含绘制图形的PIL Image对象。
    """
    # 创建一个白色背景的图像
    width, height = 400, 300
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)

    # 绘制几个简单的几何图形
    # 红色的圆
    draw.ellipse((50, 50, 150, 150), fill='red', outline='black')
    # 蓝色的正方形
    draw.rectangle((200, 50, 300, 150), fill='blue', outline='black')
    # 绿色的三角形
    draw.polygon([(100, 200), (150, 250), (50, 250)], fill='green', outline='black')

    # (可选) 添加标签
    try:
        # 使用一个常见的字体，大小适中
        # 注意：在某些环境下可能需要指定完整字体路径
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        # 如果找不到指定字体，则使用Pillow的默认字体
        font = ImageFont.load_default()

    # 绘制文本标签
    draw.text((60, 160), "Red Circle", fill="black", font=font)
    draw.text((210, 160), "Blue Square", fill="black", font=font)
    draw.text((55, 260), "Green Triangle", fill="black", font=font)

    # --- 主要改动在这里 ---
    # 移除了 image.save(...) 这一行
    
    # 直接返回内存中的Image对象
    return image

async def test_call_embedding_model():
    """
    测试调用embedding模型的方法
    """
    # 测试参数
    model_id = configured_model_id("EMBEDDING")
    embed_input_texts = ["苹果", "香蕉"]
    emb = await model_client.call_embedding_model(
        model_id=model_id, texts=embed_input_texts, batch_size=1, is_query=False,
    )

    print(f"测试开始，使用模型: {model_id}")
    print(f"输入文本: {embed_input_texts}")
    print("=" * 50)
    print("\n向量列表: ", emb)
    print("\n测试结束")

async def test_call_llm_model():
    """
    测试调用LLM模型的方法
    """
    # 测试参数
    model_id = configured_model_id("LLM")
    test_prompt = "hello"

    print(f"测试开始，使用模型: {model_id}")
    print(f"输入提示: {test_prompt}")
    print("=" * 50)

    answer = await model_client.get_llm_answer(model_id, test_prompt)
    assert answer
    print(answer)

async def test_call_vlm_model():
    """
    测试调用VLM模型的方法
    """
    # 测试参数
    model_id = configured_model_id("VLM")
    image = create_basic_shapes_image()


    print(f"测试开始，使用模型: {model_id}")
    print("=" * 50)
    response = await model_client.get_vlm_answer(model_id, image, prompt="描述该图片")
    print(f"模型响应: {response}")


# 运行测试
if __name__ == "__main__":

    asyncio.run(test_call_llm_model())
