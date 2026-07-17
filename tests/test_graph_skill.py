import uuid
import asyncio
import sys
import importlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Any
from loguru import logger

from utils.lang_detect import detect_user_language


# Add both project root and backend directory to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# 导入底层的统一图检索服务
from services.search.graph_search import GraphBaseSearch
# 导入你项目中的规范定义
from core.dictionary import PacketType
from agent.common import ContextMemory, SkillExecutionContext
module_path = "skills.skill_libs.ask-graph-skill"
ask_graph_module = importlib.import_module(module_path)
AskGraphSkill = ask_graph_module.AskGraphSkill


def build_mock_context_memory(
    user_id: str,
    agent_id: int,
    kb_id: int,
    question: str,
    standalone_query: str,
    vertex_names: list[str],
    search_top_k: int = 5,
    max_depth: int = 2,
    graph_weight: float = 1.5,
    security_level: int = 9
) -> ContextMemory:
    """
    根据 TypedDict 规范严格构造 ContextMemory 容器数据
    """
    # 1. 组装单个 Skill 的当前执行快照空间
    current_execution: SkillExecutionContext = {
        "skill": "ask-graph-skill",
        "task_description": "根据实体词：{vertex_names} 游走图谱空间，召回关联文本切片。",
        "resolved_input": question,
        "resolved_params": {
            "kb_id": kb_id,
            "vertex_names": vertex_names,
            "search_top_k": search_top_k,
            "max_depth": max_depth,
            "graph_weight": graph_weight
        },
        "start_time": datetime.now(timezone.utc),
        "end_time": None,
        "status": "RUNNING",
        "output": None,
        "output_var": "graph_results",
        "error": None
    }

    # 2. 完整拼装上下文总线记忆体
    context_memory: ContextMemory = {
        # --- 1. 基础元数据 (Session Basics) ---
        "user_id": user_id,
        "session_id": f"session_{uuid.uuid4().hex}",
        "agent_id": agent_id,
        "question": question,
        "standalone_query": standalone_query,
        "llm_model": "gpt-4o",
        "security_level": security_level,
        "tags": ["graph-test", "nexus-cube-core"],

        # --- 2. 决策快照 (Routing & Intent) ---
        "intent_context": {
            "intent": "knowledge_graph_search",
            "confidence": 0.95,
            "workflow_id": f"wf_{uuid.uuid4().hex[:6]}"
        },

        # --- 3. 控制平面 (Execution Plan) ---
        "runtime_plan": None,  # 测试阶段暂不绑定复杂的完整 Plan 对象
        "current_step_index": 0,
        "current_execution": current_execution,
        "execution_history": [],

        # --- 4. 变量中心 (The Variables Registry) ---
        "variables": {
            "kb_id": kb_id,
            "graph_results": []  # 预留给总线回填的槽位
        },

        # --- 5. 数据沉淀 (Data Buffers) ---
        "doc_results": [],
        "sql_results": [],
        "graph_results": [],  # 快捷数据出口槽位

        # --- 6. 持久化与 UI 展现 (Persistence & Streaming) ---
        "session_state": {
            "preferred_depth": max_depth
        },
        "blocks": [],

        # --- 7. 语言信息 ---
        "user_language": detect_user_language(question),

        # --- 8. 瞬时空间 (Ephemeral Space) ---
        "temp": {}
    }

    return context_memory


async def run_final_graph_skill_test():
    """
    真实驱动 AskGraphSkill 的流式全链路对齐测试函数
    """
    logger.info("⚙️ 开始加载基于规范化 TypedDict 上下文的图检索集成测试...")

    # 1. 模拟上层编排分配的真实业务参数（对齐原实验B的真实存量数据）
    mock_kb_id = 10001  # 已转化为符合底层结构的整型知识库ID
    target_entities = ["样品温度", "霍尔测量", "磁场强度", "霍尔因子"]
    user_query = "在不同样品温度下进行霍尔测量时，磁场强度和霍尔因子之间有什么具体的拓扑关联？"
    rewritten_query = "样品温度 霍尔测量 磁场强度 霍尔因子 关联性分析"

    # 2. 动用工厂函数生产标准的 ContextMemory 结构体
    context = build_mock_context_memory(
        user_id="developer_chris",
        agent_id=11,
        kb_id=11,
        question=user_query,
        standalone_query=rewritten_query,
        vertex_names=target_entities,
        search_top_k=5,
        max_depth=2,
        graph_weight=1.5
    )

    # 3. 初始化待测试的自治图谱组件
    skill = AskGraphSkill()
    
    logger.warning(f"🚀 异步流式总线建立成功。开始读取数据包。会话ID: {context['session_id']}")
    
    packet_count = 0
    captured_results = []

    try:
        # 4. 真实消费流式生成器数据
        async for packet in skill.run_stream(context=context):
            packet_count += 1
            p_type = packet.get("type")
            p_content = packet.get("content")

            if p_type == PacketType.THOUGHT:
                # 打印模型或者组件的实时中间思考链路
                logger.info(f"💭 [总线流式包 - THOUGHT] -> {p_content.strip()}")

            elif p_type == PacketType.GRAPH_RESULTS:
                # 成功拦截到处理完成并洗干净的图谱结构化记录
                captured_results = p_content
                logger.success(f"🎯 [总线流式包 - GRAPH_RESULTS] -> 成功拦截。本次召回规范文本记录数: {len(captured_results)}")

            elif p_type == PacketType.ERROR:
                # 捕获业务边界或数据库方言层抛回的致命缺陷包
                logger.error(f"❌ [总线流式包 - ERROR] -> 接收到异常阻断信号: {p_content}")

            else:
                # 捕获其余生命周期或者 UI 渲染辅助数据包
                logger.debug(f"📦 [总线流式包 - {p_type}] -> 收到瞬时快照数据片。")

        # 5. 校验流管道闭合后的上下文状态回填情况
        logger.info("🏁 异步数据通道已安全关闭。开始针对 Context 内存回填状态进行断言审计...")
        
        # 审计变量中心 (Variables Registry) 是否同步落盘
        variables_registry = context.get("variables", {})
        registered_output = variables_registry.get("graph_results", [])
        
        # 审计快捷缓冲入口数据 (Data Buffers)
        buffer_output = context.get("graph_results", [])

        logger.info(f"📊 审计快照：流输出数={len(captured_results)} | 变量中心数={len(registered_output)} | 快捷缓冲数={len(buffer_output)}")

        if len(registered_output) == len(captured_results) and len(buffer_output) == len(captured_results):
            logger.success("🎉 全链路破冰成功！AskGraphSkill 完美兼容全新 TypedDict 记忆容器，数据回填总线断言 100% 通过。")
        else:
            logger.error("⚠️ 状态机异常：流式管道输出数量与 Context 上下文落盘存储的数量不一致，请检查回填机制。")

    except Exception as e:
        logger.critical(f"💥 真实集成调用期间遭遇系统级崩溃: {str(e)}", exc_info=True)


# 方便你直接在本地测试入口或异步任务调度器中唤醒
if __name__ == "__main__":
    # 启动异步测试循环
    asyncio.run(run_final_graph_skill_test())