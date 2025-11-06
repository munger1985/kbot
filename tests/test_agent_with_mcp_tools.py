import asyncio
import sys
from pathlib import Path
from loguru import logger

# Add both project root and backend directory to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Use absolute imports from project root
from services.chat.mcp_chat import Agent
from mcp_tools import *


async def test_agent_with_mcp_tools():
    """测试Agent的MCP工具调用功能"""
    
    # 1. 创建Agent实例
    agent = Agent(agent_id=23, security=3, tags=[])
    
    try:
        # 6. 测试完整的chat流程
        print("\n🚀 测试完整chat流程")
        print("=" * 50)
        
        question = "文艺复兴是什么时候?"
        # question = "查找机器学习的最新发展"
        # question = "计算一下5的平方根乘以3的对数是多少"
        
        

        print(f"\n💬 测试问题: {question}")
        try:
            # 调用完整的chat方法
            results = await agent.chat(question)
            
            if results:
                print(f"✅ 成功获取 {len(results)} 个结果")
                for i, result in enumerate(results[:2], 1):  # 只显示前2个结果
                    print(f"  结果 {i}: {result.kb_results[0].content[:100]}...")
            else:
                print("❌ 未获取到结果")
                    
        except Exception as e:
            print(f"❌ chat流程失败: {e}")
            logger.error(f"chat流程测试失败: {e}")
        
    except Exception as e:
        print(f"💥 测试过程中发生严重错误: {e}")
        logger.error(f"测试失败: {e}")
        raise


async def main():
    """主测试函数"""
    
    # 配置日志
    logger.add("mcp_test.log", rotation="10 MB", retention="7 days", level="DEBUG")
    
    print("🚀 开始MCP工具调用系统测试")
    
    try:
        # 运行Agent测试
        await test_agent_with_mcp_tools()
        
        print("\n🎊 所有测试完成！")
        
    except Exception as e:
        print(f"💥 测试执行失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    # 运行测试
    exit_code = asyncio.run(main())
    exit(exit_code)