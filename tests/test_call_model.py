import sys
import asyncio
from pathlib import Path
from decimal import Decimal
# Add both project root and backend directory to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from utils.model_client import CallModel



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
    测试调用reranker模型的方法（已集成Qwen3格式优化）
    """
    # rerank_model_id = 61  # Qwen3-RANKER
    rerank_model_id = 24  # BGE-RANKER
    
    
    question = "猫粮怎么选？要适合幼猫的。"
    inputs_list = [
        "标题：幼猫猫粮选购要点 \n\n内容：幼猫需要高蛋白、高营养的猫粮。要选专门标注“幼猫粮”的产品，注意看蛋白质含量是否高于30%。最好含有DHA帮助大脑发育，钙磷比要均衡。每天喂3-4次，少量多餐。",
        "标题：十大猫粮品牌推荐 \n\n内容：根据成分、口碑、价格对比了市面主流猫粮品牌。皇家、渴望、爱肯拿、素力高评分较高。进口粮质量稳定但贵，国产粮性价比高。无论选哪个品牌，都要看配料表前三位是不是肉类。",
        "内容：猫咪不能吃的食物清单。巧克力、洋葱、葡萄、牛奶、酒精对猫有毒。很多人类食物对猫有害，喂食前一定要查清楚。如果猫误食了危险食物，要立即送医。",
        "内容：如何给猫洗澡。先把猫的指甲剪好，用温水慢慢淋湿，用宠物专用香波。洗完后马上用毛巾擦干，用吹风机低温吹干。整个过程要快，避免猫着凉。",
        "内容：狗狗训练基本方法。用零食作为奖励，当狗狗做出正确动作时立即给予奖励。每天训练10-15分钟，保持耐心。基本指令包括“坐下”、“握手”、“过来”。",
    ]

    

    # 调用接口 (完全不变)
    response = await CallModel().call_reranker_model(
        rerank_model_id,
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

async def validate_relevance_index():
    """验证相关类别索引"""
    from microservices.reranker.model.qwen3_reranker import Qwen3Reranker, Qwen3RerankerConfig
    
    model_config = Qwen3RerankerConfig(
                    provider="local",
                    model_name="Qwen/Qwen3-Reranker-0.6B",
                    model_path="/home/chris/models/reranker/Qwen3-Reranker-0.6B",
                    device=None,
                    max_tokens=8192,
                    batch_size=1,
                    use_fp16=False,
                    use_flash_attention=False,
                    instruction=None
                )
    
    reranker = Qwen3Reranker(model_config)
    await reranker.startup()
    test_cases = [
        ("高度相关", "招聘数据工程师", "我需要招聘一名数据工程师，要求掌握Python和SQL"),
        ("不相关", "招聘数据工程师", "我想学习钢琴演奏技巧"),
    ]
    
    print("验证相关类别索引")
    print("=" * 60)
    
    for name, query, doc in test_cases:
        score = await reranker._process_single_document(query, doc)
        
        # 需要查看内部logits来验证
        # 临时修改_process_single_document来输出更多信息
        print(f"{name}:")
        print(f"  查询: {query}")
        print(f"  文档: {doc[:50]}...")
        print(f"  分数: {score:.4f}")
        print()
    
    print("分析建议:")
    print("-" * 60)
    print("1. 如果'高度相关'文档分数 > 0.7，当前索引正确")
    print("2. 如果'高度相关'文档分数 < 0.3，可能需要切换索引")
    print("3. 最可靠的方法：查看官方模型文档或示例代码")

# 运行测试
if __name__ == "__main__":

    asyncio.run(test_call_reranker_model())
