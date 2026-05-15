from datetime import datetime
from typing import Any, TypedDict


class TaskStep(TypedDict):
    """任务执行步骤"""
    step_id: int
    skill: str
    task_description: str
    output_var: str
    condition: str | None

class ExecutionPlan(TypedDict):
    """执行计划"""
    thought: str  # 核心：必须包含思考过程
    steps: list[TaskStep]
    final_goal: str
    plan_type: str  # "workflow" (人工定义) 或 "dynamic" (LLM 生成)
    workflow_id: str | None  # 如果是 workflow 模式，记录对应的 ID
    inputs: dict[str, Any]

class SkillExecutionContext(TypedDict):
    """单个Skill的执行上下文"""
    skill: str
    task_description: str           # 原始任务描述（含占位符）
    resolved_input: str              # 解析后的实际输入
    resolved_params: dict[str, Any]  # 解析后的多维纯净业务参数字典（供具体 Skill 开箱即用）
    start_time: datetime | None
    end_time: datetime | None
    status: str
    output: Any | None
    output_var: str | None
    error: str | None

class ContextMemory(TypedDict):
    # --- 1. 基础元数据 (Session Basics) ---
    user_id: str
    session_id: str
    agent_id: int
    question: str                # 原始输入
    standalone_query: str         # LLM 改写后的意图清晰的问题
    llm_model: str
    security_level: int
    tags: list[str] | None  # 用于分类和检索的标签

    # --- 2. 决策快照 (Routing & Intent) ---
    # 由 IntentRouter 填充。示例: {"intent": "business", "workflow_id": "wf_123", "confidence": 0.9}
    intent_context: dict[str, Any] 

    # --- 3. 控制平面 (Execution Plan) ---
    # 存放当前正在运行的 ExecutionPlan 对象（包含 TaskSteps 列表）
    runtime_plan: ExecutionPlan | None      
    # 追踪当前步骤：current_execution 存放当前 Step 的实时状态
    current_step_index: int  # 记录当前执行到 runtime_plan.steps 的第几个位置
    current_execution: SkillExecutionContext | None
    # 存放已完成步骤的快照（用于反思、回溯或总结）
    execution_history: list[SkillExecutionContext]

    # --- 4. 变量中心 (The Variables Registry) ---
    # 这是实现“反查”、“动态传参”的核心。
    # 所有的 output_var 都会存入这里。
    # 示例: {"line_id": "A1", "abnormal_scores": [0.9, 0.8], "db_record_count": 5}
    variables: dict[str, Any]

    # --- 5. 数据沉淀 (Data Buffers) ---
    # 为了方便 AI 总结，依然保留 RAG 和 SQL 的结果快捷入口
    doc_results: list[dict[str, Any]]
    sql_results: list[dict[str, Any]]
    
    # --- 6. 持久化与 UI 展现 (Persistence & Streaming) ---
    session_state: dict[str, Any] # 跨会话的长期记忆（如用户偏好）
    blocks: list[dict[str, Any]]   # 前端渲染流，存储 Thought, Call, Answer, Chart 等

    # --- 7. 瞬时空间 (Ephemeral Space) ---
    temp: dict[str, Any]          # 仅限单个 Skill 内部使用的垃圾袋