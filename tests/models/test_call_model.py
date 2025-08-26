import sys
from pathlib import Path
# Add both project root and backend directory to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
from utils.call_models import CallModel

import asyncio
from decimal import Decimal

async def test_call_embedding_model():
    """
    测试调用embedding模型的方法
    """
    # 测试参数
    #embed_model_unique_name = "KBOT1/BGE-M3" # KBOT1/E5-LARGE-V2
    embed_model_unique_name = "KBOT1/OCI-Embedding"
    embed_input_texts = ["苹果", "香蕉"]
    topk = 4
    emb = await CallModel().call_embedding_model(
        embed_model_unique_name, 
        embed_input_texts
    )
    print(f"测试开始，使用模型: {embed_model_unique_name}")
    print(f"输入文本: {embed_input_texts}")
    print("=" * 50)
    print("\n向量列表: ", emb)
    print("\n测试结束")

async def test_call_llm_model():
    """
    测试调用LLM模型的方法
    """
    # 测试参数
    model_name = "KBOT1/OCI-GROK4-II"
    # model_name = "KBOT1/xai.grok-4"
    test_prompt = "文艺复兴是什么"
    
    print(f"测试开始，使用模型: {model_name}")
    print(f"输入提示: {test_prompt}")
    print("=" * 50)
    
    try:
        # 调用方法1：基本调用（流式）
        print("\n测试1：基本流式调用")
        async for chunk in CallModel().call_llm_model(model_name, test_prompt):
            print(chunk, end="", flush=True)  # 实时打印响应
        
        # 调用方法2：带额外参数
        # print("\n\n测试2：带额外参数调用")
        # async for chunk in call_llm_model(
        #     model_name, 
        #     test_prompt,
        #     temperature=0.7,
        #     max_tokens=100,
        #     top_p=Decimal('0.9')  # 测试Decimal参数转换
        # ):
        #     print(chunk, end="", flush=True)
            
        # 调用方法3：非流式模式（需要修改call_llm_model方法支持）
        # print("\n\n测试3：非流式调用")
        # result = await call_llm_model(
        #     model_name,
        #     test_prompt,
        #     stream=False
        # )
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
    rerank_model_unique_name = "KBOT1/BGE-RANKER"
    # rerank_model_unique_name = "KBOT1/JINA-RANKER"
    # rerank_model_unique_name = "KBOT1/cohere-reranker"
    question = "招聘数据工程师"
    inputs_list = [
        "<|im_end|>你好，我想要找一份有关数据科学的数据集。",
        "<|im_end|>我想做些有关数据科学的工作。",
        "<|im_end|>我想要找一份与量子计算相关的工作。",
        "<|im_end|>我的研究兴趣是量子计算，我想寻找一份相关专业的工作。"
    ]
    
    # 添加重试逻辑
    max_retries = 2
    for attempt in range(max_retries):
        try:
            rerank = await CallModel().call_reranker_model(
                rerank_model_unique_name,
                question,
                inputs_list,
                2
            )
            print(f"测试开始，使用模型: {rerank_model_unique_name}")
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
    model_unique_name = "KBOT1/Qwen-VL-MAX"
    prompt_unique_name = "KBOT1/pdf_parsing"
    image = "/mnt/f/docs/test_small.jpg"


    print(f"测试开始，使用模型: {model_unique_name}")
    print(f"输入提示的唯一标识: {prompt_unique_name}")
    print("=" * 50)
    response = await CallModel().call_vlm_model_for_parsing_picture(model_unique_name,prompt_unique_name, image)
    print(f"模型响应: {response}")

# 运行测试
if __name__ == "__main__":

    asyncio.run(test_call_reranker_model())
