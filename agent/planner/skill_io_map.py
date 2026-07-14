"""
Skill I/O 映射表 — 用于依赖分析和并行执行调度。

定义每个 Skill 读写的数据缓冲区（doc_results, sql_results, graph_results 等），
用于：
1. PlanValidator: 校验变量依赖是否可满足
2. ExecutionScheduler: 计算步骤间的隐式依赖，生成并行波次

新增 Skill 时在此处补充映射即可。
"""

SKILL_IO_MAP: dict[str, dict[str, list[str]]] = {
    # ── 知识检索类 (写入 doc_results) ──
    "ask-doc-skill": {
        "writes": ["doc_results"],
        "reads": [],
    },
    "AskDocSkill": {
        "writes": ["doc_results"],
        "reads": [],
    },

    # ── 数据查询类 (写入 sql_results) ──
    "ask-data-skill": {
        "writes": ["sql_results"],
        "reads": [],
    },
    "AskDataSkill": {
        "writes": ["sql_results"],
        "reads": [],
    },

    # ── 图谱检索类 (写入 graph_results) ──
    "ask-graph-skill": {
        "writes": ["graph_results"],
        "reads": [],
    },
    "AskGraphSkill": {
        "writes": ["graph_results"],
        "reads": [],
    },

    # ── 综合分析类 (读取所有缓冲区) ──
    "reasoning-skill": {
        "writes": [],
        "reads": ["doc_results", "sql_results", "graph_results"],
    },
    "ReasoningSkill": {
        "writes": [],
        "reads": ["doc_results", "sql_results", "graph_results"],
    },

    # ── 可视化类 (读取 sql_results) ──
    "echarts-skill": {
        "writes": [],
        "reads": ["sql_results"],
    },
    "EchartsSkill": {
        "writes": [],
        "reads": ["sql_results"],
    },

    # ── 闲聊类 (无 I/O) ──
    "chit-chat-skill": {
        "writes": [],
        "reads": [],
    },
    "ChitChatSkill": {
        "writes": [],
        "reads": [],
    },

    # ── OPS 运维类 ──
    "db-metric-skill": {
        "writes": ["metric_results", "monitor_results"],
        "reads": [],
    },
    "db-analysis-skill": {
        "writes": [],
        "reads": ["metric_results", "monitor_results", "doc_results"],
    },
    "ops-heal-skill": {
        "writes": [],
        "reads": ["doc_results", "metric_results"],
    },
}


def get_skill_io(skill_name: str) -> dict[str, list[str]]:
    """获取指定 Skill 的 I/O 特征，未注册则返回空 dict"""
    return SKILL_IO_MAP.get(skill_name, {"writes": [], "reads": []})
