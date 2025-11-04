import asyncio
import sys
from pathlib import Path
from loguru import logger

# Add both project root and backend directory to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Use absolute imports from project root
from services.chat.mcp_chat import Agent

async def test_agent_with_mcp_tools():
    """测试Agent的MCP工具调用功能"""
    
    # 1. 创建Agent实例
    agent = Agent(agent_id=1, security=0, tags=[])
    
    try:
        # 2. 测试不同类型的问题，验证工具选择
        
        test_cases = [
            {
                "question": "什么是机器学习？",
                "expected_tools": ["knowledge_base_search"],  # 期望使用知识库搜索
                "description": "知识查询类问题"
            },
            {
                "question": "计算一下2的10次方是多少？",
                "expected_tools": ["calculator"],  # 期望使用计算器
                "description": "数学计算类问题"
            },
            {
                "question": "搜索最新的AI新闻",
                "expected_tools": ["internet_search"],  # 期望使用网络搜索
                "description": "实时信息查询"
            },
            {
                "question": "帮我查找Python编程的相关资料并计算一下学习时间",
                "expected_tools": ["knowledge_base_search", "calculator"],  # 期望多工具组合
                "description": "复合型问题"
            }
        ]
        
        print("=" * 60)
        print("开始MCP工具调用功能测试")
        print("=" * 60)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📝 测试用例 {i}: {test_case['description']}")
            print(f"❓ 问题: {test_case['question']}")
            print(f"🎯 期望工具: {test_case['expected_tools']}")
            
            try:
                # 3. 调用工具选择方法
                context = {
                    "security": 0,
                    "tags": ["test"],
                    "processed_question": test_case['question']
                }
                
                tool_calls = await agent._call_llm_for_tool_selection(
                    question=test_case['question'],
                    context=context
                )
                
                # 4. 验证结果
                actual_tools = [tc.tool_name for tc in tool_calls]
                print(f"✅ 实际选择的工具: {actual_tools}")
                
                # 检查是否选择了期望的工具
                if set(test_case['expected_tools']).issubset(set(actual_tools)):
                    print("✅ 工具选择符合预期")
                else:
                    print(f"❌ 工具选择不符合预期，期望: {test_case['expected_tools']}, 实际: {actual_tools}")
                
                # 5. 显示工具调用的详细信息
                if tool_calls:
                    print("🔧 工具调用详情:")
                    for j, tool_call in enumerate(tool_calls, 1):
                        print(f"  工具 {j}: {tool_call.tool_name}")
                        print(f"    参数: {tool_call.parameters}")
                        print(f"    描述: {tool_call.description}")
                        
                        # 验证参数格式
                        if tool_call.parameters:
                            print(f"    参数验证: ✅ 参数格式正确")
                        else:
                            print(f"    参数验证: ⚠️ 无参数或参数为空")
                
                else:
                    print("⚠️  未选择任何工具")
                    
            except Exception as e:
                print(f"❌ 测试用例 {i} 执行失败: {e}")
                logger.error(f"测试用例 {i} 失败: {e}")
            
            print("-" * 50)
        
        # 6. 测试完整的chat流程
        print("\n🚀 测试完整chat流程")
        print("=" * 50)
        
        chat_test_questions = [
            "什么是深度学习？",
            "帮我计算一下(15 + 27) * 3 等于多少？",
            "查找机器学习的最新发展"
        ]
        
        for question in chat_test_questions:
            print(f"\n💬 测试问题: {question}")
            try:
                # 调用完整的chat方法
                results = await agent.chat(question)
                
                if results:
                    print(f"✅ 成功获取 {len(results)} 个结果")
                    for i, result in enumerate(results[:2], 1):  # 只显示前2个结果
                        print(f"  结果 {i}: {result.content[:100]}...")
                else:
                    print("❌ 未获取到结果")
                    
            except Exception as e:
                print(f"❌ chat流程失败: {e}")
                logger.error(f"chat流程测试失败: {e}")
        
        # 7. 测试工具执行
        print("\n🔧 测试工具执行功能")
        print("=" * 50)
        
        # 手动创建工具调用进行测试
        test_tool_calls = [
            ToolCall(
                tool_type=MCPToolType.CALCULATOR,
                tool_name="calculator",
                parameters={"expression": "2**10 + 5*3"},
                description="执行数学计算"
            ),
            ToolCall(
                tool_type=MCPToolType.KB_SEARCH,
                tool_name="knowledge_base_search",
                parameters={"query": "人工智能", "search_type": "hybrid", "limit": 5},
                description="搜索知识库"
            )
        ]
        
        for tool_call in test_tool_calls:
            print(f"\n测试执行工具: {tool_call.tool_name}")
            print(f"参数: {tool_call.parameters}")
            
            try:
                # 执行工具
                tool_result = await agent.tool_registry.execute_tool(
                    tool_call.tool_name, 
                    tool_call.parameters
                )
                
                if tool_result.success:
                    print(f"✅ 工具执行成功")
                    print(f"   结果: {tool_result.content}")
                    if tool_result.metadata:
                        print(f"   元数据: {tool_result.metadata}")
                else:
                    print(f"❌ 工具执行失败: {tool_result.error}")
                    
            except Exception as e:
                print(f"❌ 工具执行异常: {e}")
        
        print("\n" + "=" * 60)
        print("🎉 MCP工具调用功能测试完成")
        print("=" * 60)
        
    except Exception as e:
        print(f"💥 测试过程中发生严重错误: {e}")
        logger.error(f"测试失败: {e}")
        raise

async def test_llm_service_tool_support():
    """测试LLM服务的工具调用支持"""
    
    print("\n🧪 测试LLM服务工具调用支持")
    print("=" * 50)
    
    try:
        # 创建测试用的工具定义
        test_tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "执行数学计算",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "数学表达式"
                            }
                        },
                        "required": ["expression"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "knowledge_base_search",
                    "description": "搜索知识库",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "搜索查询"
                            },
                            "search_type": {
                                "type": "string",
                                "enum": ["vector", "fulltext", "hybrid"],
                                "description": "搜索类型"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "结果数量限制"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
        
        # 测试LLM服务调用
        from your_llm_service_module import LLMService  # 替换为实际的导入路径
        
        llm_service = LLMService()
        await llm_service.initialize()
        
        # 测试带工具调用的请求
        test_messages = [
            {"role": "user", "content": "计算一下2的8次方是多少？"}
        ]
        
        print("测试LLM服务工具调用...")
        response = await llm_service.chat(
            model_id=1,  # 使用合适的模型ID
            messages=test_messages,
            tools=test_tools,
            tool_choice="auto",
            stream=False
        )
        
        print("✅ LLM服务工具调用测试成功")
        if hasattr(response, 'choices') and response.choices:
            message = response.choices[0].message
            if hasattr(message, 'tool_calls') and message.tool_calls:
                print(f"检测到 {len(message.tool_calls)} 个工具调用")
                for tool_call in message.tool_calls:
                    print(f"  工具: {tool_call.function.name}")
                    print(f"  参数: {tool_call.function.arguments}")
            else:
                print("未检测到工具调用，返回直接回答")
                print(f"回答: {message.content}")
        
        await llm_service.shutdown()
        
    except Exception as e:
        print(f"❌ LLM服务工具调用测试失败: {e}")
        logger.error(f"LLM服务测试失败: {e}")

async def main():
    """主测试函数"""
    
    # 配置日志
    logger.add("mcp_test.log", rotation="10 MB", retention="7 days", level="DEBUG")
    
    print("🚀 开始MCP工具调用系统测试")
    
    try:
        # 运行Agent测试
        await test_agent_with_mcp_tools()
        
        # 运行LLM服务测试
        await test_llm_service_tool_support()
        
        print("\n🎊 所有测试完成！")
        
    except Exception as e:
        print(f"💥 测试执行失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    # 运行测试
    exit_code = asyncio.run(main())
    exit(exit_code)