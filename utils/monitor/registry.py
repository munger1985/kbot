"""
统一指标注册中心 (Unified Metric Registry)。

负责加载 YAML 配置文件，将 LLM 输出的通用 metric_code
动态渲染为对应监控工具的具体查询语句 (PromQL / Zabbix Key / ...)。

设计理念:
  - 阶段一 (语义匹配): LLM 拆解出 metric_code → 完全与底层工具解耦
  - 阶段二 (执行体翻译): 根据 monitor_type 动态渲染具体查询语句
"""

import os
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


class UnifiedMetricRegistry:
    """
    统一指标注册中心。

    加载 metrics_mapping.yaml，提供:
      - get_metric_info(): 获取指标定义（含 PromQL 模板、描述、tags）
      - render_query(): 根据 monitor_type 渲染最终查询语句
      - list_for_llm_prompt(): 列出所有注册的指标（供 LLM Prompt 注入）
    """

    def __init__(self, config_path: str | None = None):
        """
        初始化注册中心。

        Args:
            config_path: metrics_mapping.yaml 的路径。默认从 configuration/ 目录加载。
        """
        if config_path is None:
            # 默认路径: 项目根目录/configuration/metrics_mapping.yaml
            project_root = Path(__file__).parent.parent.parent
            config_path = str(project_root / "configuration" / "metrics_mapping.yaml")

        self.config_path = config_path
        self.registry: dict[str, Any] = {}

        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    self.registry = yaml.safe_load(f) or {}
                logger.info(
                    f"[UnifiedMetricRegistry] 加载完成 | 路径: {config_path} "
                    f"| 已注册指标数: {len(self.registry)}"
                )
            except Exception as e:
                logger.error(
                    f"[UnifiedMetricRegistry] 加载配置文件失败: {config_path} | 错误: {e}"
                )
                self.registry = {}
        else:
            logger.warning(
                f"[UnifiedMetricRegistry] 配置文件不存在: {config_path} "
                f"| 将以空注册表启动，所有指标查询将降级"
            )

    def get_metric_info(self, metric_code: str) -> dict[str, Any] | None:
        """
        获取指定指标编码的完整定义。

        Returns:
            包含 name, description, tags, prometheus_template 等的字典。
            未注册时返回 None。
        """
        info = self.registry.get(metric_code)
        if not info:
            logger.warning(f"[UnifiedMetricRegistry] 未注册的指标编码: {metric_code}")
            return None
        return dict(info)

    def render_query(
        self,
        metric_code: str,
        monitor_type: str,
        db_type: str,
        params: dict[str, Any] | None = None,
    ) -> str:
        """
        根据 monitor_type 和 db_type 将指标模板渲染为最终查询语句。

        Args:
            metric_code:  统一指标编码 (如 "db_cpu_utilization")
            monitor_type: 监控工具类型 ("prometheus" / "zabbix")
            db_type:      目标数据库类型 ("oracle" / "postgresql" / "mysql" / ...)
            params:       动态参数 (如 {"instance": "192.168.1.50:9161"})

        Returns:
            渲染后的查询语句 (PromQL / Zabbix Key / ...)

        Raises:
            ValueError: 指标未注册 / 不支持该 monitor_type / 不支持该 db_type
        """
        info = self.get_metric_info(metric_code)
        if info is None:
            raise ValueError(f"未注册的指标编码: {metric_code}")

        params = params or {}

        if monitor_type == "prometheus":
            # 优先使用多源异构 providers 结构
            providers = info.get("providers")
            if providers and isinstance(providers, dict):
                template = providers.get(db_type)
                if not template:
                    supported = list(providers.keys())
                    raise ValueError(
                        f"指标 [{metric_code}] 不支持数据库类型 [{db_type}]。"
                        f"支持的数据库: {supported}"
                    )
            else:
                # 向后兼容: 旧的扁平 prometheus_template 字段
                template = info.get("prometheus_template")
                if not template:
                    raise ValueError(
                        f"指标 [{metric_code}] 未配置 Prometheus 查询模板"
                    )

            try:
                return template.format(**params)
            except KeyError as e:
                missing_key = str(e).strip("'")
                raise ValueError(
                    f"渲染 PromQL 模板失败，缺失参数: {missing_key}。"
                    f"模板: {template}，已提供参数: {list(params.keys())}"
                ) from e

        elif monitor_type == "zabbix":
            # Zabbix 暂未实现，走预留路径
            providers = info.get("providers")
            if providers and isinstance(providers, dict):
                item_key = providers.get(db_type) if isinstance(providers, dict) else None
            else:
                item_key = info.get("zabbix_item_key")
            if not item_key:
                raise NotImplementedError(
                    f"指标 [{metric_code}] 未配置 Zabbix Item Key，"
                    f"且 Zabbix 驱动暂未实现"
                )
            try:
                return item_key.format(**params)
            except KeyError as e:
                missing_key = str(e).strip("'")
                raise ValueError(
                    f"渲染 Zabbix Key 模板失败，缺失参数: {missing_key}"
                ) from e

        else:
            raise ValueError(f"不支持的监控工具类型: {monitor_type}")

    def list_for_llm_prompt(
        self,
        monitor_type: str = "prometheus",
        db_type: str | None = None,
    ) -> str:
        """
        将所有已注册指标格式化为 LLM Prompt 注入文本。

        供 OpsTaskPlanner 在生成执行计划时，让 LLM 知道有哪些可用指标。

        Args:
            monitor_type: 只列出支持该类型的指标
            db_type:      可选，只列出支持该数据库类型的指标

        Returns:
            格式化的指标列表文本，可直接注入 Prompt
        """
        lines = []
        for code, info in self.registry.items():
            # 检查 providers 结构
            providers = info.get("providers")

            # 按 db_type 过滤
            if db_type and providers and isinstance(providers, dict):
                if db_type not in providers:
                    continue

            # 全局过滤: 检查是否支持目标监控类型
            if monitor_type == "prometheus":
                has_template = bool(
                    (providers and isinstance(providers, dict) and any(providers.values()))
                    or info.get("prometheus_template")
                )
                if not has_template:
                    continue
            elif monitor_type == "zabbix":
                if not info.get("zabbix_item_key"):
                    continue

            name = info.get("name", code)
            desc = info.get("description", "")
            tags = info.get("tags", [])
            params_info = ""
            if info.get("required_params"):
                params_info = f"，可用参数: {', '.join(info['required_params'])}"

            lines.append(
                f"- **{code}** ({name}): {desc}{params_info} | 标签: {', '.join(tags)}"
            )

        if not lines:
            return "（暂无可用监控指标）"

        return "\n".join(lines)

    def get_metrics_by_tag(self, tag: str) -> list[str]:
        """按标签筛选指标编码列表"""
        result = []
        for code, info in self.registry.items():
            if tag in info.get("tags", []):
                result.append(code)
        return result

    def __len__(self) -> int:
        return len(self.registry)

    def __contains__(self, metric_code: str) -> bool:
        return metric_code in self.registry
