import asyncio
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("KBOT_RUN_MODEL_SMOKE") != "1",
    reason="需要已启动并配置真实模型的四类推理服务",
)

# 加载 .env 文件（必须在导入任何使用配置的模块之前）
from dotenv import load_dotenv
env_path = Path(__file__).resolve().parent.parent / ".env"
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
    embed_model_name = "KBOT_Qwen3-Embedding-4B"
    embed_input_texts = ["苹果", "香蕉"]
    topk = 4
    kwargs = {}
    kwargs['batch_size'] = 1
    kwargs["served_model_name"] = embed_model_name
    kwargs['texts'] = embed_input_texts
    kwargs['is_query'] = False
    emb = await model_client.call_embedding_model(**kwargs)

    print(f"测试开始，使用模型: {embed_model_name}")
    print(f"输入文本: {embed_input_texts}")
    print("=" * 50)
    print("\n向量列表: ", emb)
    print("\n测试结束")

async def test_call_llm_model():
    """
    测试调用LLM模型的方法
    """
    # 测试参数
    model_name = "KBOT_DeepSeek-Chat"
    test_prompt = "hello"

    print(f"测试开始，使用模型: {model_name}")
    print(f"输入提示: {test_prompt}")
    print("=" * 50)

    try:
        # # 调用方法1：基本调用（流式）
        # print("\n测试1：基本流式调用")
        # async for chunk in model_client.call_llm_model(model_name, test_prompt):
        #     print(chunk, end="", flush=True)  # 实时打印响应

        # 调用方法2：带额外参数
        # print("\n\n测试2：带额外参数调用")
        # async for chunk in model_client.call_llm_model(
        #     model_name,
        #     test_prompt,
        #     temperature=0.7,
        #     max_tokens=100,
        #     top_p=Decimal('0.9')  # 测试Decimal参数转换
        # ):
        #     print(chunk, end="", flush=True)

        # 调用方法3：非流式模式（需要修改call_llm_model方法支持）
        print("\n\n测试3：非流式调用")
        async for chunk in model_client.call_llm_model(
            model_name,
            test_prompt,
            stream=False,
            max_tokens=10
        ):
            result = chunk
        print(result)

    except Exception as e:
        print(f"\n测试失败: {str(e)}")
    finally:
        print("\n\n测试结束")

async def test_call_reranker_model():
    """
    测试调用reranker模型的方法（已集成Qwen3格式优化）
    """
    rerank_model_name = "KBOT_Qwen3-Reranker-4B"


    question = "猫粮怎么选？要适合幼猫的。"
    inputs_list = [
        "标题：幼猫猫粮选购要点 \n\n内容：幼猫需要高蛋白、高营养的猫粮。要选专门标注「幼猫粮」的产品，注意看蛋白质含量是否高于30%。最好含有DHA帮助大脑发育，钙磷比要均衡。每天喂3-4次，少量多餐。",
        "标题：十大猫粮品牌推荐 \n\n内容：根据成分、口碑、价格对比了市面主流猫粮品牌。皇家、渴望、爱肯拿、素力高评分较高。进口粮质量稳定但贵，国产粮性价比高。无论选哪个品牌，都要看配料表前三位是不是肉类。",
        "内容：猫咪不能吃的食物清单。巧克力、洋葱、葡萄、牛奶、酒精对猫有毒。很多人类食物对猫有害，喂食前一定要查清楚。如果猫误食了危险食物，要立即送医。",
        "内容：如何给猫洗澡。先把猫的指甲剪好，用温水慢慢淋湿，用宠物专用香波。洗完后马上用毛巾擦干，用吹风机低温吹干。整个过程要快，避免猫着凉。",
        "内容：狗狗训练基本方法。用零食作为奖励，当狗狗做出正确动作时立即给予奖励。每天训练10-15分钟，保持耐心。基本指令包括「坐下」、「握手」、「过来」。",
    ]



    # 调用接口 (完全不变)
    response = await model_client.call_reranker_model(
        rerank_model_name,
        question,
        inputs_list,
        top_k=5
    )
    print(response)  # 应该看到索引4（数据工程师）排在第一位


async def test_call_vlm_model():
    """
    测试调用VLM模型的方法
    """
    # 测试参数
    model_name = "KBOT_QwenVL"
    image = create_basic_shapes_image()


    print(f"测试开始，使用模型: {model_name}")
    print("=" * 50)
    response = await model_client.get_vlm_answer(model_name, image, prompt="描述该图片")
    print(f"模型响应: {response}")


# 运行测试
if __name__ == "__main__":

    asyncio.run(test_call_llm_model())
