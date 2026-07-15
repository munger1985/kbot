from datetime import datetime
from typing import Any, Literal, TypedDict

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
    """
    全域通用 - 单个 Skill 的核心运行时执行上下文/审计快照
    兼容 Business 动态变量渲染流 与 Ops 挂起审批控制链
    """
    skill: str                       # 调用的 Skill 注册名称 (e.g., "db-metric-skill", "knowledge-query")
    task_description: str            # 静态编排时的原始任务描述（可能包含 {{variable}} 占位符）
    resolved_input: str              # 经过运行时引擎动态渲染、变量替换后的实际文本输入
    resolved_params: dict[str, Any]  # 经过大模型提取或框架绑定后的多维纯净结构化参数字典（供具体 Skill 内部开箱即用）
    start_time: datetime | None      # 技能物理启动时间
    end_time: datetime | None        # 技能物理结束时间
    
    # 物理状态：引入 Ops 的悬挂状态，完美支持"人工审批/自愈熔断"拦截机制
    status: Literal["pending", "running", "suspended", "success", "failed", "blocked"]
    
    output: Any | None               # 技能执行的最终结算实体数据（如 SQL 结果集、RAG 召回块）
    output_var: str                  # 执行完毕后，指定将 output 挂载到 context.variables 的哪个 Key 中
    error: str | None                # 发生异常崩溃时的完整堆栈或错误描述错误快照
    answer: str | None               # 步骤执行过程中累积的回答文本（HITL 挂起时记录）