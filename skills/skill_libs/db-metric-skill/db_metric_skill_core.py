# skills/skill_libs/db-metric-skill/db_metric_skill_core.py
"""
DBMetricSkill v2 — Prometheus 优先 · 专家 SQL 兜底。

架构升级:
  v1 (旧): 自然语言 -> Embedding 向量匹配 -> ops_metric_sql 表 -> 执行 SQL
  v2 (新): 自然语言 -> LLM 拆解为 metric_code + params
           -> 第一阶段: 匹配 Prometheus 指标 -> 渲染 PromQL -> HTTP API -> 时序数据
           -> 第二阶段: 如需深度诊断 -> LLM 单选题选专家工具 -> 直连数据库执行 SQL

设计原则:
  - 90% 常规指标走 Prometheus（安全、快速、无数据库负载）
  - 10% 深度诊断走预埋的 16 个专家 SQL（LLM Function Calling 精准路由）
  - Zabbix 和 OEM 作为备选监控数据源，通过 monitor_type 切换
"""

import json
import inspect
from typing import Any, AsyncGenerator
from loguru import logger

from skills import BaseSkill, SkillMeta, SkillDomain, SkillRunMode
from agent.common.ops_context import OpsContextMemory
from agent.common.diagnostic_tools import DatabaseDiagnosticTools
from core.dictionary import PacketType
from agent.prompt import default_prompt
from core.config.settings import get_prompt_config
from utils.clients import AIModelClient, OpsDBExecutor
from utils.monitor import PrometheusClient, ZabbixProvider, OEMProvider, UnifiedMetricRegistry


class DBMetricSkill(BaseSkill):
    """
    DB 智能运维指标听诊器 v2:
    监控优先: 通过 Prometheus API 获取常规时序指标。
    诊断兜底: 当需要深度根因分析时, 通过 LLM 单选题机制精准调用 16 个专家诊断 SQL 工具。
    """

    meta = SkillMeta(
        name="db-metric-skill",
        description="【数据库运维专有工具 v2】Prometheus 优先获取常规监控指标, 16 个专家 SQL 工具兜底深度根因诊断",
        domain=SkillDomain.OPS,
        run_mode=SkillRunMode.READ_ONLY,
    )

    def __init__(self):
        super().__init__()
        self.model_client = AIModelClient()
        self.db_executor = OpsDBExecutor()

    async def run_stream(
        self, context: OpsContextMemory, **kwargs
    ) -> AsyncGenerator[dict[str, Any], None]:
        """DB 运维指标核心流 v2: Prometheus 监控优先 -> 专家 SQL 工具兜底。"""
        curr_exec = context.get("current_execution") or {}
        task_desc = (
            curr_exec.get("resolved_input")
            or curr_exec.get("task_description")
            or kwargs.get("resolved_input")
            or kwargs.get("task_description")
            or ""
        )

        # 从 ctx.variables 中获取基础设施引用（由 OpsOrchestrator 注入）
        prometheus_client: PrometheusClient | None = context.get("variables", {}).get("_prometheus_client")
        zabbix_client: ZabbixProvider | None = context.get("variables", {}).get("_zabbix_client")
        oem_client: OEMProvider | None = context.get("variables", {}).get("_oem_client")
        metric_registry: UnifiedMetricRegistry | None = context.get("variables", {}).get("_metric_registry")
        ops_db_executor: OpsDBExecutor | None = context.get("variables", {}).get("_ops_db_executor")

        # 回退到自行初始化
        if prometheus_client is None:
            prometheus_client = PrometheusClient()
        if zabbix_client is None:
            zabbix_client = ZabbixProvider()
        # OEM 客户端由 OpsOrchestrator 按需注入，不在此 fallback（避免默认 localhost 连接浪费）
        if metric_registry is None:
            metric_registry = UnifiedMetricRegistry()
        if ops_db_executor is None:
            ops_db_executor = OpsDBExecutor()

        instance_id = context["instance_id"]
        db_type = context["db_type"]
        monitor_type = context.get("monitor_type", "prometheus")
        prometheus_label = context.get("prometheus_instance_label")
        zabbix_host_name = context.get("zabbix_host_name")
        oem_target_name = context.get("oem_target_name")
        oem_target_type = context.get("oem_target_type", "oracle_database")
        llm_model = context["llm_model"]

        logger.info(
            f"[OpsTrack v2] DBMetricSkill 启动 | 实例: {instance_id} | "
            f"引擎: {db_type} | 监控源: {monitor_type} | 任务: {task_desc[:80]}"
        )

        if not instance_id or not db_type:
            yield {"type": PacketType.ERROR, "content": "❌ 运维上下文缺失关键参数: instance_id 或 db_type, 技能强行熔断。"}
            return

        # =====================================================================
        # 阶段一: 监控指标查询（按 monitor_type 选择数据源）
        # =====================================================================
        metric_code, extracted_params = await self._resolve_metric_code(
            task_desc, metric_registry, db_type, llm_model, monitor_type
        )

        if metric_code and monitor_type == "prometheus":
            yield {"type": PacketType.THOUGHT, "content": f"📊 正在通过 Prometheus 查询指标: [{metric_code}]...\n"}

            try:
                render_params = {"instance": prometheus_label or instance_id}
                if prometheus_label:
                    logger.debug(
                        f"[OpsTrack v2] 使用 CMDB 配置的 prometheus_label: {prometheus_label}"
                    )
                else:
                    logger.warning(
                        f"[OpsTrack v2] CMDB 未配置 prometheus_instance_label, "
                        f"降级使用 instance_id={instance_id} 作为 Prometheus instance 标签. "
                        f"如果 Prometheus 中 instance 标签值为其他格式（如 host:port），查询将返回空。"
                    )
                if extracted_params:
                    # 禁止 LLM 覆盖 instance 标签：instance 必须来自 CMDB 配置
                    extracted_params.pop("instance", None)
                    render_params.update(extracted_params)

                promql = metric_registry.render_query(metric_code, "prometheus", db_type, render_params)
                logger.info(f"[OpsTrack v2] db_type={db_type} PromQL: {promql}")

                monitor_result = await prometheus_client.query_instant(promql)
                monitor_result.metric_code = metric_code

                logger.info(
                    f"[OpsTrack v2] Prometheus 查询成功 | metric={metric_code} "
                    f"| instance_label={render_params.get('instance')} "
                    f"| series_count={len(monitor_result.series)} "
                    f"| sample={monitor_result.series[:1]}"
                )

                # 诊断：如果指定 instance 没数据，不加过滤查一次看指标是否存在
                if len(monitor_result.series) == 0:
                    metric_name = promql.split("{")[0] if "{" in promql else promql
                    diag_promql = f"{metric_name}"
                    try:
                        diag_result = await prometheus_client.query_instant(diag_promql)
                        instances = list({
                            s.get("labels", {}).get("instance", "?")
                            for s in diag_result.series[:10]
                        }) if diag_result.series else []
                        logger.warning(
                            f"[OpsTrack v2] 指标 {metric_code} 查询返回空。"
                            f"不带 label 过滤查询 ({diag_promql}): "
                            f"total={len(diag_result.series)}, "
                            f"可用 instance 值: {instances}"
                        )
                    except Exception:
                        pass

                prometheus_has_data = len(monitor_result.series) > 0

                summary_text = self._format_monitor_result(metric_code, monitor_result)
                logger.debug(
                    f"[OpsTrack v2] 即将 yield MONITOR_RESULTS | "
                    f"data_len={len(monitor_result.series)}, meta_keys={list(monitor_result.series[0].keys()) if monitor_result.series else 'empty'}"
                )
                yield {
                    "type": PacketType.MONITOR_RESULTS,
                    "content": {
                        "data": monitor_result.series,
                        "meta": {
                            "metric_code": metric_code,
                            "source": "prometheus",
                            "promql": promql,
                            "summary": summary_text,
                        },
                    },
                }

                # Prometheus 有数据时，让 LLM 判断是否需要补充 DB 诊断
                if prometheus_has_data:
                    should_query_db = await self._should_supplement_with_db(
                        metric_code=metric_code,
                        series=monitor_result.series,
                        task_desc=task_desc,
                        db_type=db_type,
                        llm_model=llm_model,
                    )
                    if should_query_db:
                        logger.info(
                            f"[OpsTrack v2] LLM 判定 Prometheus 数据不足，"
                            f"继续尝试专家 SQL 工具箱补充深度诊断..."
                        )
                        # 继续执行阶段二
                    else:
                        logger.info(
                            f"[OpsTrack v2] LLM 判定 Prometheus 数据已足够，跳过数据库查询"
                        )
                        return
                else:
                    return  # Prometheus 无数据，不重复尝试诊断工具（阶段二会接管）

            except ValueError as e:
                logger.warning(f"[OpsTrack v2] Prometheus 查询失败, 降级: {e}")
                yield {"type": PacketType.WARNING, "content": f"⚠️ Prometheus 指标查询未命中: {e}, 尝试专家 SQL 工具箱..."}
            except ConnectionError as e:
                logger.error(f"[OpsTrack v2] Prometheus 连接失败: {e}")
                yield {"type": PacketType.WARNING, "content": f"⚠️ Prometheus Server 不可达 ({e}), 降级到专家 SQL 工具箱..."}
            except Exception as e:
                logger.error(f"[OpsTrack v2] Prometheus 查询异常: {e}")
                yield {"type": PacketType.WARNING, "content": f"⚠️ Prometheus 查询异常: {e}, 尝试专家 SQL 工具箱..."}

        elif metric_code and monitor_type == "zabbix":
            yield {"type": PacketType.THOUGHT, "content": f"📊 正在通过 Zabbix 查询指标: [{metric_code}]...\n"}

            try:
                render_params = {"instance": zabbix_host_name or instance_id}
                if zabbix_host_name:
                    logger.debug(
                        f"[OpsTrack v2] 使用 CMDB 配置的 zabbix_host_name: {zabbix_host_name}"
                    )
                else:
                    logger.warning(
                        f"[OpsTrack v2] CMDB 未配置 zabbix_host_name, "
                        f"降级使用 instance_id={instance_id} 作为 Zabbix host 名称. "
                        f"如果 Zabbix 中主机名为其他值，查询将返回空。"
                    )
                if extracted_params:
                    extracted_params.pop("instance", None)
                    render_params.update(extracted_params)

                item_key = metric_registry.render_query(metric_code, "zabbix", db_type, render_params)
                logger.info(f"[OpsTrack v2] db_type={db_type} Zabbix Item Key: {item_key}")

                monitor_result = await zabbix_client.query_instant(item_key)
                monitor_result.metric_code = metric_code

                logger.info(
                    f"[OpsTrack v2] Zabbix 查询成功 | metric={metric_code} "
                    f"| host={render_params.get('instance')} "
                    f"| series_count={len(monitor_result.series)}"
                )

                zabbix_has_data = len(monitor_result.series) > 0

                summary_text = self._format_monitor_result(metric_code, monitor_result)
                yield {
                    "type": PacketType.MONITOR_RESULTS,
                    "content": {
                        "data": monitor_result.series,
                        "meta": {
                            "metric_code": metric_code,
                            "source": "zabbix",
                            "item_key": item_key,
                            "summary": summary_text,
                        },
                    },
                }

                if zabbix_has_data:
                    should_query_db = await self._should_supplement_with_db(
                        metric_code=metric_code,
                        series=monitor_result.series,
                        task_desc=task_desc,
                        db_type=db_type,
                        llm_model=llm_model,
                    )
                    if should_query_db:
                        logger.info(
                            f"[OpsTrack v2] LLM 判定 Zabbix 数据不足，"
                            f"继续尝试专家 SQL 工具箱补充深度诊断..."
                        )
                    else:
                        logger.info(
                            f"[OpsTrack v2] LLM 判定 Zabbix 数据已足够，跳过数据库查询"
                        )
                        return
                else:
                    return

            except ValueError as e:
                logger.warning(f"[OpsTrack v2] Zabbix 查询失败, 降级: {e}")
                yield {"type": PacketType.WARNING, "content": f"⚠️ Zabbix 指标查询未命中: {e}, 尝试专家 SQL 工具箱..."}
            except ConnectionError as e:
                logger.error(f"[OpsTrack v2] Zabbix 连接失败: {e}")
                yield {"type": PacketType.WARNING, "content": f"⚠️ Zabbix Server 不可达 ({e}), 降级到专家 SQL 工具箱..."}
            except Exception as e:
                logger.error(f"[OpsTrack v2] Zabbix 查询异常: {e}")
                yield {"type": PacketType.WARNING, "content": f"⚠️ Zabbix 查询异常: {e}, 尝试专家 SQL 工具箱..."}

        elif metric_code and monitor_type == "oem":
            yield {"type": PacketType.THOUGHT, "content": f"📊 正在通过 Oracle Enterprise Manager 查询指标: [{metric_code}]...\n"}

            if not oem_client:
                yield {"type": PacketType.WARNING, "content": "⚠️ OEM 客户端未配置或不可用，跳过 OEM 查询"}
                return

            try:
                render_params = {"target": oem_target_name or instance_id}
                if oem_target_name:
                    logger.debug(
                        f"[OpsTrack v2] 使用 CMDB 配置的 oem_target_name: {oem_target_name}"
                    )
                else:
                    logger.warning(
                        f"[OpsTrack v2] CMDB 未配置 oem_target_name, "
                        f"降级使用 instance_id={instance_id} 作为 OEM target 名称. 如果 OEM 目标名称为其他值，查询将返回空。"
                    )
                if extracted_params:
                    extracted_params.pop("instance", None)
                    extracted_params.pop("target", None)
                    render_params.update(extracted_params)

                oem_query = metric_registry.render_query(metric_code, "oem", db_type, render_params)
                logger.info(f"[OpsTrack v2] db_type={db_type} OEM Query: {oem_query}")

                monitor_result = await oem_client.query_instant(oem_query)
                monitor_result.metric_code = metric_code

                logger.info(
                    f"[OpsTrack v2] OEM 查询成功 | metric={metric_code} "
                    f"| target={render_params.get('target')} "
                    f"| series_count={len(monitor_result.series)}"
                )

                oem_has_data = len(monitor_result.series) > 0

                summary_text = self._format_monitor_result(metric_code, monitor_result)
                yield {
                    "type": PacketType.MONITOR_RESULTS,
                    "content": {
                        "data": monitor_result.series,
                        "meta": {
                            "metric_code": metric_code,
                            "source": "oem",
                            "oem_query": oem_query,
                            "summary": summary_text,
                        },
                    },
                }

                if oem_has_data:
                    should_query_db = await self._should_supplement_with_db(
                        metric_code=metric_code,
                        series=monitor_result.series,
                        task_desc=task_desc,
                        db_type=db_type,
                        llm_model=llm_model,
                    )
                    if should_query_db:
                        logger.info(
                            f"[OpsTrack v2] LLM 判定 OEM 数据不足，"
                            f"继续尝试专家 SQL 工具箱补充深度诊断..."
                        )
                    else:
                        logger.info(
                            f"[OpsTrack v2] LLM 判定 OEM 数据已足够，跳过数据库查询"
                        )
                        return
                else:
                    return

            except ValueError as e:
                logger.warning(f"[OpsTrack v2] OEM 查询失败, 降级: {e}")
                yield {"type": PacketType.WARNING, "content": f"⚠️ OEM 指标查询未命中: {e}, 尝试专家 SQL 工具箱..."}
            except ConnectionError as e:
                logger.error(f"[OpsTrack v2] OEM 连接失败: {e}")
                yield {"type": PacketType.WARNING, "content": f"⚠️ OEM Server 不可达 ({e}), 降级到专家 SQL 工具箱..."}
            except Exception as e:
                logger.error(f"[OpsTrack v2] OEM 查询异常: {e}")
                yield {"type": PacketType.WARNING, "content": f"⚠️ OEM 查询异常: {e}, 尝试专家 SQL 工具箱..."}

        # =====================================================================
        # 阶段二: 专家 SQL 工具箱（Prometheus 有数据时作为补充，无数据时作为兜底）
        # =====================================================================
        yield {"type": PacketType.THOUGHT, "content": "🔧 正在通过专家诊断工具箱进行深度数据库探查...\n"}

        try:
            tool_result = await self._invoke_diagnostic_tool(
                task_desc=task_desc,
                db_type=db_type,
                db_executor=ops_db_executor,
                llm_model=llm_model,
                instance_id=instance_id,
                extracted_params=extracted_params,
            )

            if tool_result is None:
                yield {
                    "type": PacketType.WARNING,
                    "content": "⚠️ 未能匹配到合适的专家诊断工具。请提供更明确的运维指令。",
                }
                return

            tool_name, exec_result = tool_result

            yield {
                "type": PacketType.METRIC_RESULTS,
                "content": {
                    "data": exec_result,
                    "meta": {
                        "tool_name": tool_name,
                        "description": f"专家诊断工具 [{tool_name}] 执行结果",
                        "source": "diagnostic_tool",
                    },
                },
            }

        except Exception as sql_err:
            logger.error(f"专家诊断工具执行崩溃: {str(sql_err)}")
            yield {"type": PacketType.ERROR, "content": f"❌ 专家诊断脚本执行崩溃: {str(sql_err)}"}

    # ========================================================================
    # 私有辅助方法
    # ========================================================================

    async def _resolve_metric_code(
        self,
        task_desc: str,
        registry: UnifiedMetricRegistry,
        db_type: str,
        llm_model: str,
        monitor_type: str = "prometheus",
    ) -> tuple[str | None, dict[str, Any] | None]:
        """通过 LLM 将自然语言任务匹配到 Prometheus 指标编码。"""
        if len(registry) == 0:
            return None, None

        metrics_prompt = registry.list_for_llm_prompt(monitor_type=monitor_type, db_type=db_type)

        prompt = await default_prompt.generate(
            get_prompt_config().ops_metric_matching,
            task_desc=task_desc,
            metrics_list=metrics_prompt,
        )
        try:
            result = await self.model_client.get_llm_json(llm_model, prompt)
            code = result.get("metric_code")
            params = result.get("params") or {}

            if code and code in registry:
                logger.info(f"[OpsTrack v2] LLM 匹配指标: {code} | 参数: {params}")
                return code, params
            else:
                logger.info("[OpsTrack v2] LLM 未能匹配任何监控指标")
                return None, None
        except Exception as e:
            logger.warning(f"[OpsTrack v2] LLM 指标匹配失败: {e}")
            return None, None

    async def _invoke_diagnostic_tool(
        self,
        task_desc: str,
        db_type: str,
        db_executor: OpsDBExecutor,
        llm_model: str,
        instance_id: str,
        extracted_params: dict[str, Any] | None = None,
    ) -> tuple[str, list[dict[str, Any]]] | None:
        """让 LLM 从 16 个专家诊断工具中做单选题, 然后执行选中的工具。"""
        tools_manifest = DatabaseDiagnosticTools.get_tool_manifest()

        prompt = await default_prompt.generate(
            get_prompt_config().ops_diagnostic_tool,
            task_desc=task_desc,
            db_type=db_type,
            tools_manifest=tools_manifest,
        )
        try:
            llm_choice = await self.model_client.get_llm_json(llm_model, prompt)
            tool_name = llm_choice.get("tool_name")
            args = llm_choice.get("arguments") or {}

            if not tool_name:
                logger.info("[OpsTrack v2] LLM 认为无需使用任何诊断工具")
                return None

            logger.info(f"[OpsTrack v2] LLM 选择工具: {tool_name} | 参数: {args}")

            tools = DatabaseDiagnosticTools(db_type=db_type, db_executor=db_executor, instance_id=instance_id)
            tool_method = getattr(tools, tool_name, None)

            if tool_method is None:
                logger.error(f"[OpsTrack v2] 工具不存在: {tool_name}")
                return None

            sig = inspect.signature(tool_method)
            filtered_args = {k: v for k, v in args.items() if k in sig.parameters}

            result = await tool_method(**filtered_args)
            return tool_name, result

        except Exception as e:
            logger.error(f"[OpsTrack v2] 诊断工具调用失败: {e}")
            raise

    async def _should_supplement_with_db(
        self,
        metric_code: str,
        series: list[dict],
        task_desc: str,
        db_type: str,
        llm_model: str,
    ) -> bool:
        """让 LLM 判断 Prometheus 数据是否足够，还是需要补充数据库诊断"""
        if len(series) == 0:
            return True  # 没数据，肯定要查库

        # 取前 3 条数据样本，去掉 __ 系统标签
        sample = []
        for s in series[:3]:
            labels = {k: v for k, v in s.get("labels", {}).items()
                      if not k.startswith("__")}
            sample.append({
                "value": s.get("value", "N/A"),
                "labels": labels,
            })

        prompt = await default_prompt.generate(
            get_prompt_config().ops_metric_supplement,
            task_desc=task_desc,
            db_type=db_type,
            metric_code=metric_code,
            series_count=len(series),
            sample_json=json.dumps(sample, ensure_ascii=False, indent=2),
        )

        try:
            resp = await self.model_client.get_llm_json(
                model_name=llm_model,
                prompt=prompt,
                temperature=0,
            )
            decision = str(resp.get("decision", resp.get("answer", "YES"))).strip().upper()
            return decision.startswith("Y")
        except Exception:
            # LLM 调用失败，保守处理：需要查库
            return True

    def _format_monitor_result(self, metric_code: str, result) -> str:
        """将 Prometheus MetricResult 格式化为可读摘要"""
        info = {}
        if hasattr(result, 'series') and result.series:
            info = result.series[0] if result.series else {}

        value = info.get("value", "N/A")
        labels = info.get("labels", {})

        return f"指标 [{metric_code}] 当前值: {value} | 标签: {labels}"
