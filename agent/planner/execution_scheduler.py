"""
执行调度器 (Execution Scheduler)

基于依赖图将步骤分组为执行波次 (waves)。
同一波次内的步骤无相互依赖，可异步并行执行。
波次之间严格串行，保证数据一致性。
"""

from agent.common.skill_context import TaskStep


def compute_execution_waves(
    steps: list[TaskStep],
    dependency_graph: dict[int, set[int]],
) -> list[list[int]]:
    """
    使用 Kahn 算法将步骤分组为拓扑有序的执行波次。

    同一波次内的步骤满足:
    - 彼此之间无直接或间接依赖
    - 所有前置波次的步骤已完成

    Args:
        steps: 执行计划的步骤列表
        dependency_graph: PlanValidator 生成的依赖图 {step_index: {依赖的step_index集合}}

    Returns:
        波次列表 [[step_idx, step_idx], [step_idx], ...]
        每个元素是一个波次，包含该波次中可并行执行的步骤索引

    Example:
        步骤: [查故障码, 查方案文档, 查生产指标, 综合分析, 画图]
        依赖: 方案文档→故障码, 生产指标→故障码, 综合分析→方案文档+生产指标, 画图→综合分析

        Wave 0: [0]           (查故障码, 无依赖)
        Wave 1: [1, 2]        (查方案文档, 查生产指标 — 彼此独立, 可并行)
        Wave 2: [3]           (综合分析)
        Wave 3: [4]           (画图)
    """
    if not steps:
        return []

    # 计算入度（每个步骤有多少个未完成的前置依赖）
    in_degree: dict[int, int] = {}
    for i in range(len(steps)):
        in_degree[i] = len(dependency_graph.get(i, set()))

    waves: list[list[int]] = []
    remaining = dict(in_degree)

    while remaining:
        # 当前波次：入度为 0 的所有步骤
        wave = [i for i, deg in remaining.items() if deg == 0]
        if not wave:
            # 不应发生（对于无环图），但防御性编程
            break
        waves.append(wave)

        # 移除当前波次，更新被这些步骤阻塞的后续步骤的入度
        for i in wave:
            del remaining[i]

        # 重新计算入度：检查被波次中步骤阻塞的步骤
        for j in list(remaining.keys()):
            new_deg = 0
            for dep in dependency_graph.get(j, set()):
                if dep in remaining:
                    new_deg += 1
            remaining[j] = new_deg

    return waves


def get_step_dependencies(steps: list[TaskStep], step_index: int, dep_graph: dict[int, set[int]]) -> set[int]:
    """获取某个步骤的所有直接前置依赖"""
    return dep_graph.get(step_index, set())
