from datetime import datetime
from typing import Any, Literal, TypedDict
from .skill_context import ExecutionPlan, SkillExecutionContext

class OpsContextMemory(TypedDict):
    # =========================================================================
    # --- 1. 基础运维元数据 (Ops Session Basics) ---
    # =========================================================================
    trace_id: str                  # 运维全链路追踪 ID（串联告警、自愈流程、审计日志）
    user_id: str                   # 发起运维指令的用户 ID，用于记忆检索与审计追溯
    session_id: str                # 对齐业务线：前端单次长连接的会话 ID
    agent_id: int                  # 执行此任务的专职运维 Agent ID（如 "oracle-dba-agent"）
    trigger_type: Literal["manual", "webhook", "cron"] # 触发源：人工提问、监控系统回调、定时巡检
    command_or_query: str          # 原始输入的运维指令或告警摘要
    client_time: str               # 客户端请求时间
    client_tz: str                 # 客户端时区
    llm_model: str                 # 编排与参数提取所使用的大模型名称
    embedding_model: str           # 嵌入模型名称，用于文本相似度计算

    # =========================================================================
    # --- 2. 目标拓扑与内核实体 (Target Topology & Kernel Info) ---
    # =========================================================================
    instance_id: str               # 物理/云端实例唯一ID (e.g., "ins-oracle-prod-01")
    instance_name: str             # 物理/云端实例名称（如 "oracle-prod-01"）
    db_type: str                   # 数据库类型（如 "oracle" "mysql" "postgresql"）
    version_code: int              # 精准到内核版本的数字代码 (e.g., 26000000)
    db_role: Literal["primary", "standby", "cluster_node"] # 节点角色，防止 Planner 跑到备库去执行高危变动
    environment: Literal["prod", "stg", "dev"] # 环境隔离标签：如果是 prod，自动收紧安全策略并开启审批门禁

    # =========================================================================
    # --- 2.1 监控数据源配置 (Monitoring Source Config) ---
    # =========================================================================
    monitor_type: str              # 监控数据源类型: "prometheus" | "zabbix" | "oem" | "none"
    prometheus_instance_label: str | None  # Prometheus 中该实例的 instance 标签值
    zabbix_host_name: str | None  # Zabbix 监控主机名称
    oem_target_name: str | None   # Oracle Enterprise Manager 中的目标名称
    oem_target_type: str | None   # OEM 目标类型（如 "oracle_database"）

    # =========================================================================
    # --- 3. 告警事件快照 (Alert Snapshot) ---
    # =========================================================================
    alert_context: dict[str, Any] | None

    # =========================================================================
    # --- 4. 统一自愈控制平面 (Execution Plan Control Plane) ---
    # =========================================================================
    runtime_plan: ExecutionPlan | None       # 线性/条件执行蓝图
    current_step_index: int                  # 记录当前执行到 runtime_plan.steps 的第几个位置
    current_execution: SkillExecutionContext | None 
    execution_history: list[SkillExecutionContext]  

    # =========================================================================
    # --- 5. 高危动作熔断门禁 (Safety & Approval Gate) ---
    # =========================================================================
    approval_context: dict[str, Any] | None

    # =========================================================================
    # --- 6. 全局运维变量中心 (The Ops Variables Registry) ---
    # =========================================================================
    variables: dict[str, Any]

    # =========================================================================
    # --- 7. 数据沉淀区与瞬时空间 (Ops Data Buffers & Ephemeral Space) ---
    # =========================================================================
    metric_results: list[dict[str, Any]]    # 合并后的唯一指标/数据沉淀区：承载 DBMetricSkill 的探测结果、执行元数据与步骤上下文
    monitor_results: list[dict[str, Any]]   # Prometheus/Zabbix 监控数据沉淀区：标准化时序数据，供 DBAnalysisSkill 做趋势分析
    os_log_snapshots: list[str]             # 沉淀捞出来的 alert.log 或操作系统的 OOM 崩溃日志段
    doc_results: list[dict[str, Any]]       # 沉淀本轮改写检索后，真实命中的知识库 SOP 文档切片及分数（纯字典格式）

    # 瞬时空间：仅限单个 Skill 内部使用的无污染临时沙箱，随单步 Skill 销毁而清空
    temp: dict[str, Any]

    # =========================================================================
    # --- 8. HITL 人机协同 (Human-in-the-Loop) ---
    # =========================================================================
    is_resuming: bool                    # 是否从挂起状态恢复（Skill 检测到此标志时跳过充分性检查）
    hitl_history: list[dict[str, Any]]   # 多轮排查 Timeline（追加而非覆盖）