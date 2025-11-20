import sys
import asyncio
from pathlib import Path
from decimal import Decimal
# Add both project root and backend directory to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from utils.call_models import CallModel


async def test_call_embedding_model():
    """
    测试调用embedding模型的方法
    """
    # 测试参数
    # embed_model_id = 21 #"KBOT1/BGE-M3"
    # embed_model_id = 23 #"KBOT1/E5-LARGE-V2"
    # embed_model_id = 33 #"KBOT1/OCI-Embedding"
    embed_model_id = 41	# Qwen3-Embedding
    embed_input_texts = ["苹果", "香蕉"]
    topk = 4
    kwargs = {}
    kwargs['batch_size'] = 1
    kwargs['model_id'] = embed_model_id
    kwargs['texts'] = embed_input_texts
    kwargs['is_query'] = False
    emb = await CallModel().call_embedding_model(**kwargs)

    print(f"测试开始，使用模型: {embed_model_id}")
    print(f"输入文本: {embed_input_texts}")
    print("=" * 50)
    print("\n向量列表: ", emb)
    print("\n测试结束")

async def test_call_llm_model():
    """
    测试调用LLM模型的方法
    """
    # 测试参数
    # model_id = 39 #'KBOT1/OCI-cohere'
    # model_id = 40 #"KBOT1/OCI-GROK4-II"
    model_id = 22 #"KBOT1/DeepSeek V3"
    test_prompt = "hello"
    
    print(f"测试开始，使用模型: {model_id}")
    print(f"输入提示: {test_prompt}")
    print("=" * 50)
    
    try:
        # 调用方法1：基本调用（流式）
        print("\n测试1：基本流式调用")
        async for chunk in CallModel().call_llm_model(model_id, test_prompt):
            print(chunk, end="", flush=True)  # 实时打印响应
        
        # 调用方法2：带额外参数
        # print("\n\n测试2：带额外参数调用")
        # async for chunk in CallModel().call_llm_model(
        #     model_id, 
        #     test_prompt,
        #     temperature=0.7,
        #     max_tokens=100,
        #     top_p=Decimal('0.9')  # 测试Decimal参数转换
        # ):
        #     print(chunk, end="", flush=True)
            
        # 调用方法3：非流式模式（需要修改call_llm_model方法支持）
        # print("\n\n测试3：非流式调用")
        # async for chunk in CallModel().call_llm_model(
        #     model_id,
        #     test_prompt,
        #     stream=False,
        #     max_tokens=10
        # ):
        #     result = chunk
        # print(result)
        
    except Exception as e:
        print(f"\n测试失败: {str(e)}")
    finally:
        print("\n\n测试结束")

async def test_call_reranker_model():
    """
    测试调用reranker模型的方法
    """
    # 测试参数
    # rerank_model_id = 24 #"KBOT1/BGE-RANKER"
    rerank_model_id = 25 #"KBOT1/JINA-RANKER"
    # rerank_model_id = 61 # Qwen3-RANKER
    question = "招聘数据工程师"
    inputs_list = [
        "<|im_end|>你好，我想要找一份有关数据科学的数据集。",
        "<|im_end|>我想做些有关数学专业的工作。",
        "<|im_end|>我想要找一份与量子计算相关的工作。",
        "<|im_end|>我的研究兴趣是量子计算，我想寻找一份相关专业的工作。",
        "<|im_end|>你好，我想要找一份数据工程师的工作。",
    ]
    
    # 添加重试逻辑
    max_retries = 2
    for attempt in range(max_retries):
        try:
            rerank = await CallModel().call_reranker_model(
                rerank_model_id,
                question,
                inputs_list,
                5
            )
            print(f"测试开始，使用模型: {rerank_model_id}")
            print(f"结果: {rerank}")
            print("=" * 50)
            break
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"测试失败: {str(e)}")
            else:
                print(f"第 {attempt + 1} 次尝试失败，正在重试...")
                await asyncio.sleep(1)
    


async def test_call_vlm_model():
    """
    测试调用VLM模型的方法
    """
    # 测试参数
    model_id = 30 #"KBOT1/Qwen-VL-MAX"
    prompt_unique_name = "SYSTEM/image2text"
    image = "/mnt/f/docs/test_small.jpg"


    print(f"测试开始，使用模型: {model_id}")
    print(f"输入提示的唯一标识: {prompt_unique_name}")
    print("=" * 50)
    response = await CallModel().call_vlm_model_for_parsing_picture(model_id, image)
    print(f"模型响应: {response}")


async def test_call_similarity_model():
    """
    测试调用相似度模型的方法
    """
    # 测试参数
    # 测试参数
    # embed_model_id = 21 #"KBOT1/BGE-M3"
    # embed_model_id = 23 #"KBOT1/E5-LARGE-V2"
    # embed_model_id = 33 #"KBOT1/OCI-Embedding"
    embed_model_id = 41	# Qwen3-Embedding
    text1 = "你好"
    text2 = "你好吗"
    method = "cosine"

    print(f"测试开始，使用模型: {embed_model_id}")
    print(f"输入文本1: {text1}")
    print(f"输入文本2: {text2}")
    print(f"相似度计算方法: {method}")
    print("=" * 50)
    
    try:
        similarity = await CallModel().compute_similarity(embed_model_id, text1, text2, method)
        print(f"相似度: {similarity}")
    except Exception as e:
        print(f"测试失败: {str(e)}")
        print("可能的原因：")
        print("1. CUDA设备不可用或不兼容")
        print("2. 模型加载失败")
        print("3. 内存不足")
        print("4. 模型服务未启动")


# 运行测试
if __name__ == "__main__":

    asyncio.run(test_call_embedding_model())
