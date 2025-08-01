import sys
from pathlib import Path
# Add both project root and backend directory to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
from utils.call_models import call_llm_model

import asyncio
from decimal import Decimal

async def test_call_llm_model():
    """
    测试调用LLM模型的方法
    """
    # 测试参数
    model_name = "KBOT1/DeepSeek V3"
    test_prompt = "你好"
    
    print(f"测试开始，使用模型: {model_name}")
    print(f"输入提示: {test_prompt}")
    print("=" * 50)
    
    try:
        # 调用方法1：基本调用（流式）
        print("\n测试1：基本流式调用")
        async for chunk in call_llm_model(model_name, test_prompt):
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

# 运行测试
if __name__ == "__main__":
    asyncio.run(test_call_llm_model())