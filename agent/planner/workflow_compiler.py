from collections import deque, defaultdict
from typing import Any
from loguru import logger
from agent.common import TaskStep, ExecutionPlan
from skills import SkillManager

class WorkflowCompiler:
    """
    将字典格式的 Workflow 定义编译为系统可执行的 ExecutionPlan (TypedDict)

    同时提供 linearize_for_llm() 方法，将 SOP 的 DAG 结构转换为 LLM 友好的
    Markdown 摘要，供 LLMPlanner 在生成计划时作为约束参考。
    """

    def __init__(self, skill_manager: SkillManager):
        self.skill_manager = skill_manager

    def _validate_skill_name(self, raw_name: str) -> str | None:
        """
        校验技能名称是否在系统中已注册。返回规范化名称，未注册时返回 None。
        """
        if not raw_name or raw_name == "UnknownSkill":
            return None

        # 尝试精确匹配
        if raw_name in self.skill_manager._skills:
            return raw_name

        # 尝试规范化模糊匹配
        normalized = self.skill_manager._normalize_skill_name(raw_name)
        if normalized in self.skill_manager._skills:
            return normalized

        # 尝试忽略大小写的类名匹配
        for reg_name in self.skill_manager._skills:
            if reg_name.lower().replace("-", "").replace("_", "") == raw_name.lower().replace("-", "").replace("_", ""):
                return reg_name

        return None

    def compile(self, workflow_data: dict[str, Any], query: str) -> ExecutionPlan:
        # 1. 结构准备 - 直接从字典中取值
        workflow_id = str(workflow_data.get('id', ""))
        workflow_name = workflow_data.get('name', '未命名流程')
        workflow_desc = workflow_data.get('description', '未定义业务意图')
        nodes = workflow_data.get('nodes') or {}
        edges = workflow_data.get('edges') or []

        if not nodes:
            logger.warning(f"Workflow '{workflow_name}' 没有定义任何节点")
            return {
                "thought": "该流程为空，无法执行。",
                "steps": [],
                "final_goal": query,
                "plan_type": "workflow",
                "workflow_id": workflow_id,
                "inputs": {} # 初始化为空，由 Planner 进行预注入
            }

        # 2. 构建邻接表和入度表用于拓扑排序
        adj = defaultdict(list)
        in_degree = defaultdict(int)
        
        for edge in edges:
            u, v = str(edge['source']), str(edge['target'])
            if u in nodes and v in nodes:
                adj[u].append(v)
                in_degree[v] += 1

        # 3. 执行拓扑排序 (Kahn's Algorithm)
        queue = deque([node_id for node_id in nodes if in_degree[node_id] == 0])
        ordered_node_ids = []

        while queue:
            curr_id = queue.popleft()
            ordered_node_ids.append(curr_id)

            for neighbor in adj[curr_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # 4. 环路检测
        if len(ordered_node_ids) != len(nodes):
            logger.error(f"Workflow '{workflow_name}' 包含循环依赖或孤立节点")
            raise ValueError("流程编排包含循环依赖，请检查前端连线")

        # 5. 拼装 TaskStep (TypedDict) 列表（含技能名校验）
        task_steps: list[TaskStep] = []
        for index, node_id in enumerate(ordered_node_ids):
            node_info = nodes[node_id]
            inner = node_info.get('data', node_info)

            raw_skill = inner.get('implementation_key', "UnknownSkill")
            validated_skill = self._validate_skill_name(raw_skill)

            if not validated_skill:
                logger.error(
                    f"Workflow '{workflow_name}' 节点 [{node_id}] 引用了未注册的技能: '{raw_skill}'，"
                    f"该步骤将被跳过。请检查前端配置或技能包是否已部署。"
                )
                continue

            # 构造符合 TaskStep 结构的字典
            step: TaskStep = {
                "step_id": len(task_steps) + 1,
                "skill": validated_skill,
                "task_description": inner.get('instruction') or inner.get('description', ""),
                "output_var": inner.get('output_var', f"step_{len(task_steps) + 1}_result"),
                "condition": inner.get('condition')
            }
            task_steps.append(step)

        # 6. 构造 ExecutionPlan (符合 TypedDict 结构)
        thought_process = (
            f"用户发起了标准作业程序(SOP): 【{workflow_name}】。\n"
            f"业务意图：{workflow_desc}。\n"
            f"执行逻辑：系统已将编排好的 DAG 图编译为 {len(task_steps)} 个拓扑有序步骤。"
        )

        return {
            "thought": thought_process,
            "steps": task_steps,
            "final_goal": f"完成【{workflow_name}】定义的业务流程，解答：{query}",
            "plan_type": "workflow",
            "workflow_id": workflow_id,
            "inputs": {} # 初始为空字典
        }

    def linearize_for_llm(self, workflow_data: dict[str, Any]) -> list[dict[str, Any]]:
        """
        将 SOP 的 DAG 结构转换为 LLM 友好的线性步骤摘要。

        与 compile() 的区别:
        - compile() 返回完整的 ExecutionPlan (用于确定性执行)
        - linearize_for_llm() 返回轻量步骤摘要 (作为 LLM 规划的约束输入)

        每个步骤摘要包含:
        - step_id: 排序后的步骤编号
        - skill: 已校验的技能名
        - instruction: 原始的任务描述（通用性的，LLM 会结合 query 具象化）
        - output_var: 结果变量名
        """
        nodes = workflow_data.get('nodes') or {}
        edges = workflow_data.get('edges') or []

        if not nodes:
            return []

        # 拓扑排序
        adj: dict[str, list[str]] = {nid: [] for nid in nodes}
        in_degree: dict[str, int] = {nid: 0 for nid in nodes}

        for edge in edges:
            u, v = str(edge['source']), str(edge['target'])
            if u in nodes and v in nodes:
                adj[u].append(v)
                in_degree[v] += 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        ordered: list[str] = []

        while queue:
            curr = queue.pop(0)
            ordered.append(curr)
            for neighbor in adj[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # 构建步骤摘要
        summary: list[dict[str, Any]] = []
        for idx, node_id in enumerate(ordered):
            node_info = nodes[node_id]
            inner = node_info.get('data', node_info)

            raw_skill = inner.get('implementation_key', 'UnknownSkill')
            validated_skill = self._validate_skill_name(raw_skill)

            if not validated_skill:
                continue

            summary.append({
                "step_id": idx + 1,
                "skill": validated_skill,
                "instruction": inner.get('instruction') or inner.get('description', ''),
                "output_var": inner.get('output_var', f"step_{idx + 1}_result"),
            })

        return summary