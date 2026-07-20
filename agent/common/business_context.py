from typing import Any, TypedDict

from .skill_context import SkillExecutionContext, ExecutionPlan

class ContextMemory(TypedDict):
    # --- 1. 基础元数据 (Session Basics) ---
    user_id: str
    session_id: str
    agent_id: int
    question: str                # 原始输入
    standalone_query: str         # LLM 改写后的意图清晰的问题
    search_keywords: str  # 从用户输入中提取的搜索关键词
    llm_model: str # 编排与参数提取所使用的大模型名称
    embedding_model: str           #_model: str          
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
    graph_results: list[dict[str, Any]]
    
    # --- 6. 持久化与 UI 展现 (Persistence & Streaming) ---
    session_state: dict[str, Any] # 跨会话的长期记忆（如用户偏好）
    blocks: list[dict[str, Any]]   # 前端渲染流，存储 Thought, Call, Answer, Chart 等

    # --- 7. 语言信息 (Language) ---
    user_language: str            # 检测到的用户语言（如 "中文"/"English"/"日本語"/"한국어"/"हिन्दी"/"العربية"/"ไทย"/"Русский"）

    # --- 8. 瞬时空间 (Ephemeral Space) ---
    temp: dict[str, Any]          # 仅限单个 Skill 内部使用的垃圾袋