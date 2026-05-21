import asyncio
import importlib
import sys
from pathlib import Path
from datetime import datetime
from typing import Any
from loguru import logger

# Add both project root and backend directory to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.dictionary import PacketType
module_path = "skills.skill_libs.ask-graph-skill"
ask_graph_module = importlib.import_module(module_path)
AskGraphSkill = ask_graph_module.AskGraphSkill


async def main():
    logger.info("🚀 开始初始化 AskGraphSkill 真实调用测试 (直连底层服务)...")

    # 1. 初始化真实的 Skill 实例（此时会调用真实的 GraphBaseSearch() 并加载连接池）
    try:
        skill = AskGraphSkill()
    except Exception as e:
        logger.error(f"❌ 技能初始化失败，请检查数据库配置或依赖服务是否启动: {e}")
        return

    # 2. 严格遵循你给出的 ContextMemory 和 SkillExecutionContext 定义构建上下文容器
    # 模拟图片中真实的物理实验/半导体参数图谱实体
    mock_vertex_names = ["样品温度", "霍尔测量", "磁场强度", "霍尔因子"]
    
    # 假设你在系统中已经创建好的一个有效知识库 ID，请根据真实情况修改这个整数 ID
    target_kb_id = 101  

    current_execution_context: dict[str, Any] = {
        "skill": "ask-graph-skill",
        "task_description": "分析 {{vertex_names}} 之间的多度拓扑关联及连锁影响",
        "resolved_input": "样品温度, 霍尔测量, 磁场强度",
        # 严格将纯净的业务参数注入 resolved_params，完全对接你的 skill.md 定义
        "resolved_params": {
            "vertex_names": mock_vertex_names,
            "kb_id": target_kb_id,
            "search_top_k": 5,
            "max_depth": 2,
            "graph_weight": 1.5
        },
        "start_time": datetime.now(),
        "end_time": None,
        "status": "RUNNING",
        "output": None,
        "output_var": "physics_graph_output",  # 显式定义下游使用的输出变量名
        "error": None
    }

    # 构造完整的 ContextMemory
    context: dict[str, Any] = {
        # --- 1. 基础元数据 (Session Basics) ---
        "user_id": "real_engineer_user_2026",
        "session_id": "session_real_call_test",
        "agent_id": 9527,
        "question": "当样品温度和磁场强度改变时，对霍尔测量的结果有什么关联影响？",
        "standalone_query": "样品温度 磁场强度 霍尔测量 关联分析",
        "llm_model": "gpt-4o",
        "security_level": 9,
        "tags": ["physics_lab", "oracle_26ai_test"],

        # --- 2. 决策快照 (Routing & Intent) ---
        "intent_context": {"intent": "graph_topology_analysis", "confidence": 0.98}, 

        # --- 3. 控制平面 (Execution Plan) ---
        "runtime_plan": None,  # 真实测试中暂不依赖复杂的 ExecutionPlan 对象
        "current_step_index": 1,
        "current_execution": current_execution_context,
        "execution_history": [],

        # --- 4. 变量中心 (The Variables Registry) ---
        "variables": {},  # 期望运行后被回填：{"physics_graph_output": [...]}

        # --- 5. 数据沉淀 (Data Buffers) ---
        "doc_results": [],
        "sql_results": [],
        "graph_results": [],  # 期望运行后被回填的快捷入口

        # --- 6. 持久化与 UI 展现 (Persistence & Streaming) ---
        "session_state": {},
        "blocks": [],

        # --- 7. 瞬时空间 (Ephemeral Space) ---
        "temp": {}
    }

    logger.info(f"🧬 已载入真实实体过滤词: {mock_vertex_names}，准备驱动图谱游走...")
    print("-" * 80)

    # 3. 驱动真实的流式管道，打印并捕捉总线吐出的每一个 Packet 状态
    packet_counter = 0
    has_results = False
    
    try:
        async for packet in skill.run_stream(context=context):
            packet_counter += 1
            p_type = packet.get("type")
            p_content = packet.get("content")
            
            # 在控制台高亮流式交互细节
            if p_type == PacketType.THOUGHT:
                logger.info(f"[🚀 流式总线 - THOUGHT 状态包]:\n{p_content}")
            elif p_type == PacketType.GRAPH_RESULTS:
                logger.success(f"[🎉 流式总线 - GRAPH_RESULTS 结果包]: 成功拉取到分发数据明细。")
                has_results = True
            elif p_type == PacketType.DONE:
                logger.success(f"[🏁 流式总线 - DONE 单步生命周期结束包]: 收到正常终止信号。")
            elif p_type == PacketType.ERROR:
                logger.error(f"[⚠️ 流式总线 - ERROR 异常包]: 遭遇核心级阻碍: {p_content}")
                
        print("-" * 80)
        
        # 4. 核心后置校验：严格验证真实数据库跑完后，ContextMemory 内存数据和变量中心的写回结果
        logger.info("🔍 正在对运行完毕的 ContextMemory 注册中心进行状态盘点...")
        
        # 盘点 A：数据沉淀层检测
        if context["graph_results"]:
            logger.success(f"📊 [成功] 数据沉淀层 context['graph_results'] 成功捕获数据，共计 {len(context['graph_results'])} 条。")
            # 打印首条记录展示真实数据面貌
            first_record = context["graph_results"][0]
            print(f"   ↳ 首条命中关联文件: {first_record.get('title')}")
            print(f"   ↳ 关联分数 (Score): {first_record.get('score')}")
            print(f"   ↳ 动态图关系路径追踪 (search_helper): {first_record.get('search_helper')}")
        else:
            logger.warning("⚠️ 数据沉淀层 context['graph_results'] 为空，请确认 Oracle 26ai 中对应图谱是否存在这些实体的关系边。")

        # 盘点 B：变量中心动态绑定检测（解决上个版本拼写错误和回填覆盖的硬伤）
        target_var_key = context["current_execution"]["output_var"]  # "physics_graph_output"
        if target_var_key in context["variables"]:
            logger.success(f"💾 [成功] 变量注册中心已被成功击穿并回填！")
            logger.success(f"   ↳ 已在当前 Session 内注册变量: context['variables']['{target_var_key}']")
            
            # 校验变量数据与沉淀层数据是否完全等价（总线联动一致性）
            if context["variables"][target_var_key] == context["graph_results"]:
                logger.success("   ↳ [一致性校验通过] 变量中心的数据与快捷沉淀层完全等价，下游 Reasoning 引擎可通过模板无缝消费。")
        else:
            logger.error(f"❌ [失败] 变量中心缺失绑定的变量 Key: '{target_var_key}'")

    except Exception as exc:
        logger.critical(f"💥 单元测试执行流发生崩溃: {exc}", exc_info=True)


if __name__ == "__main__":
    # 使用 Python 标准异步驱动入口启动
    asyncio.run(main())