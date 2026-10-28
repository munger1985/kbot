"""AIOps 面向用户文本的统一中文展示字典。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import MappingProxyType
from typing import Any


_DISPLAY_NAMES = {
    # 数据库与资源
    "ORACLE": "Oracle",
    "MYSQL": "MySQL",
    "POSTGRESQL": "PostgreSQL",
    "ACTIVE": "启用",
    "DISABLED": "停用",
    "REVOKED": "已撤销",
    "PAUSED": "已暂停",
    "RETIRED": "已退役",
    "ARCHIVED": "已归档",
    # 运行、对话与证据充分性
    "CREATED": "已创建",
    "QUEUED": "排队中",
    "ACCEPTED": "已受理",
    "UNDERSTANDING": "正在理解问题",
    "PLANNING": "正在制定诊断计划",
    "COLLECTING": "正在采集证据",
    "ASSESSING": "正在评估证据",
    "REPLANNING": "正在补充诊断计划",
    "ANSWERING": "正在组织回答",
    "RUNNING": "进行中",
    "WAITING_INPUT": "等待补充信息",
    "WAITING_USER": "等待用户补充",
    "WAITING_APPROVAL": "等待审批",
    "PROPOSAL_PENDING": "处置建议待审批",
    "COMPLETED": "已完成",
    "PARTIAL": "部分完成",
    "FAILED": "失败",
    "CANCELLED": "已取消",
    "EXPIRED": "已过期",
    "RETRY_WAIT": "等待重试",
    "ANSWERABLE": "证据充分，可以回答",
    "NEEDS_CLARIFICATION": "需要补充说明",
    "NEEDS_EVIDENCE": "仍需补充证据",
    "CAPABILITY_UNAVAILABLE": "所需诊断能力不可用",
    "UNSAFE": "当前操作不安全",
    # 根因、发现与风险
    "CONFIRMED": "已确认",
    "PROBABLE": "很可能",
    "LIKELY": "很可能",
    "POSSIBLE": "可能",
    "INCONCLUSIVE": "证据不足，无法定论",
    "UNKNOWN": "未知",
    "CRITICAL": "严重",
    "HIGH": "高",
    "MEDIUM": "中",
    "LOW": "低",
    "INFO": "提示",
    "LOCK_WAIT": "会话阻塞",
    "LONG_SESSION": "长会话",
    "DG_LAG": "Data Guard 延迟",
    "WAIT_CLASS": "等待类异常",
    "TABLESPACE": "表空间容量异常",
    "SQL_STATS_STALE": "统计信息过期",
    "EXACHECK_FAIL": "ExaCheck 检查失败",
    "EXACHECK_WARNING": "ExaCheck 检查警告",
    "REPLICATION_LAG": "复制延迟",
    "DEAD_TUPLES": "无效元组堆积",
    "AUTOVACUUM": "自动清理异常",
    "IDLE_SESSION": "空闲会话异常",
    "CONNECTION_USAGE": "连接使用率异常",
    "QUERY_RATE": "查询速率异常",
    "INVALID_OBJECT": "无效对象",
    "ARCHIVE_HEADROOM": "归档恢复区余量不足",
    "BACKUP_FAILED": "备份失败",
    "BACKUP_MISSING": "缺少备份记录",
    "LONG_TRANSACTION": "长事务",
    "TOP_SQL": "高负载 SQL",
    "CHECKPOINT_PRESSURE": "检查点压力",
    "FREEZE_HEADROOM": "事务冻结余量不足",
    "BLOAT_CANDIDATE": "表膨胀候选",
    "ARCHIVER_FAILURE": "归档进程故障",
    "RECOVERY_CONFIGURATION": "恢复链配置风险",
    "REPLICATION_SLOT": "复制槽异常",
    "DATA_QUALITY": "数据质量",
    "ANALYZE_STALE": "统计信息需重新分析",
    # 对话输入、证据与展示类型
    "CHAT": "人工对话",
    "SITUATION": "告警情境",
    "RUN": "诊断运行",
    "REPORT": "诊断报告",
    "API": "接口调用",
    "ROOT": "根任务",
    "ALERT": "告警触发",
    "SCHEDULE": "策略计划",
    "TEXT": "文字",
    "IMAGE": "图片",
    "FILE": "文件",
    "SQL_OUTPUT": "SQL 输出",
    "COMMAND_OUTPUT": "命令输出",
    "LOG": "日志",
    "QUESTION": "问题",
    "ORACLE_ALERT_LOG": "Oracle 告警日志",
    "DATABASE_LOG": "数据库日志",
    "SQL_RESULT": "SQL 结果",
    "COMMAND_RESULT": "命令结果",
    "METRIC_SNAPSHOT": "指标快照",
    "CONFIGURATION": "配置信息",
    "SCREENSHOT": "截图",
    "MIXED": "混合材料",
    "MARKDOWN": "文字说明",
    "TABLE": "数据表格",
    "CHART": "趋势图表",
    "TABLE_AND_CHART": "数据表格与图表",
    "EVIDENCE_REFERENCES": "证据引用",
    "CLARIFICATION": "补充说明请求",
    "EVIDENCE_REQUEST": "补充证据请求",
    "PROPOSAL_SUMMARY": "处置建议摘要",
    "VERIFICATION_COMPARISON": "处置前后验证对比",
    "FINDING_CARDS": "诊断发现卡片",
    "ANALYSIS_MARKDOWN": "诊断分析",
    "SOLUTION_MARKDOWN": "解决方案",
    "FACT_CONFIRMATION": "运维事实确认",
    "REPORT_ARTIFACT_LINKS": "报告附件",
    "HTML_REPORT_LINKS": "HTML 报告附件",
    "IMPLEMENTATION_RUNBOOK": "实施操作文档",
    "SUPPORTS": "支持当前判断",
    "CONTRADICTS": "反驳当前判断",
    "CONTEXT": "背景信息",
    "FRESH": "数据新鲜",
    "STALE": "数据过期",
    # 证据口径与取证策略
    "CURRENT_ACTIVITY": "当前活动快照",
    "CUMULATIVE_SINCE_LOAD": "统计累计值（起点由数据源及重置情况决定）",
    "SNAPSHOT_DELTA": "快照区间增量",
    "HISTORICAL_SAMPLES": "历史采样",
    "NOT_APPLICABLE": "不适用",
    "SOURCE_VERIFIED": "来源已验证",
    "USER_PROVIDED": "用户提供",
    "MODEL_INFERENCE": "模型推断",
    "ADAPTIVE": "按问题选择证据来源",
    "MONITORING_FIRST": "优先使用监控数据",
    "MONITORING_UNAVAILABLE": "监控数据不可用",
    "MONITORING_FORECAST_INSUFFICIENT": "历史数据不足，无法预测",
    "MONITORING_FORECAST_LOW_CONFIDENCE": "预测可信度较低",
    "MONITORING_COVERAGE_INSUFFICIENT": "监控采样覆盖不足",
    "MONITORING_WINDOW_INSUFFICIENT": "监控时间窗口不完整",
    "MONITORING_EVIDENCE_INSUFFICIENT": "监控证据不足",
    "DATABASE_FIRST": "优先查询数据库",
    "COMBINED": "联合使用监控与数据库证据",
    "CURRENT": "当前状态",
    "HISTORICAL": "历史分析",
    "HISTORICAL_AND_FORECAST": "历史分析与趋势预测",
    # 工作流与动作
    "ALERT_DIAGNOSIS": "告警诊断",
    "INSPECTION": "巡检",
    "CHAT_TURN": "对话诊断",
    "CHANGE": "受控变更",
    "VERIFICATION": "效果验证",
    "OBSERVE": "观测",
    "DIAGNOSE": "诊断",
    "EXPLAIN": "解释",
    "ASSESS": "评估",
    "COMPARE": "对比",
    "PLAN": "制定方案",
    "VERIFY": "验证",
    "INSPECT": "巡检",
    "UNDERSTAND": "理解问题",
    "GENERAL": "通用诊断",
    "SINGLE_SQL_PERFORMANCE": "单条 SQL 性能诊断",
    "READ_ONLY_LOOKUP": "只读查询",
    "CONTROLLED_ACTION": "受控动作",
    "FULL_INVESTIGATION": "完整诊断",
    "NONE": "无",
    "ADVISORY": "仅生成处置建议",
    "EXECUTE": "申请受控执行",
    "DRAFT": "草稿",
    "PENDING": "待处理",
    "PENDING_APPROVAL": "等待审批",
    "ADVISORY_READY": "人工处置建议已就绪",
    "APPROVED": "已批准",
    "REJECTED": "已拒绝",
    "SUPERSEDED": "已被新版本替代",
    "CONSUMED": "已使用",
    "SUBMITTED": "已提交",
    "SUCCEEDED": "成功",
    "TIMED_OUT": "已超时",
    "SKIPPED": "已跳过",
    "ANSWERED": "已答复",
    "DATA_REQUIRED": "需要补充数据",
    "MANUAL_DIAGNOSTIC_SQL": "需要人工执行诊断 SQL",
    "DIAGNOSTIC_QUERY_APPROVAL": "诊断查询待审批",
    "CHANGE_APPROVAL": "受控变更待审批",
    "INCIDENT": "故障诊断报告",
    "PERFORMANCE": "性能分析报告",
    "INSPECTION_DAILY": "每日巡检报告",
    "INSPECTION_WEEKLY": "每周巡检报告",
    "INSPECTION_MONTHLY": "每月巡检报告",
    "INSPECTION_QUARTERLY": "季度巡检报告",
    "INSPECTION_ANNUAL": "年度巡检报告",
    "INSPECTION_CUSTOM": "自定义周期巡检报告",
    "COMPARISON": "处置前后对比报告",
    "CANCEL_RUN": "取消诊断运行",
    "ANSWER_HITL": "答复人工确认",
    "CANCEL_HITL": "取消人工确认",
    "APPROVE_PROPOSAL": "批准处置建议",
    "REJECT_PROPOSAL": "拒绝处置建议",
    "RECORD_MANUAL_RESULT": "登记人工处理结果",
    # DBA 领域与实施方案
    "SQL_PERFORMANCE": "SQL 性能",
    "SESSION_AND_LOCK": "会话与锁",
    "INSTANCE_PERFORMANCE": "实例性能",
    "STORAGE_AND_CAPACITY": "存储与容量",
    "BACKUP_AND_RECOVERY": "备份与恢复",
    "HIGH_AVAILABILITY": "高可用",
    "REPLICATION": "复制",
    "SECURITY_AND_PRIVILEGE": "安全与权限",
    "CONNECTION_AND_NETWORK": "连接与网络",
    "MAINTENANCE": "数据库维护",
    "PATCH_AND_UPGRADE": "补丁与升级",
    "DATA_INTEGRITY": "数据完整性",
    "ALERT_AND_LOG": "告警与日志",
    "HOST_AND_OS": "主机与操作系统",
    "ORACLE_ADG_BUILD": "Oracle Data Guard 搭建",
    "ORACLE_RAC_BUILD": "Oracle RAC 搭建",
    "ORACLE_RMAN_BACKUP_BUILD": "Oracle RMAN 备份体系搭建",
    "ORACLE_RMAN_RECOVERY": "Oracle RMAN 恢复",
    "ORACLE_RU_PATCH": "Oracle RU 补丁实施",
    "ORACLE_DATABASE_UPGRADE": "Oracle 数据库升级",
    "ORACLE_DATABASE_MIGRATION": "Oracle 数据库迁移",
    "ORACLE_CLONE_REFRESH": "Oracle 数据库克隆与刷新",
    "ORACLE_DATAPUMP_MIGRATION": "Oracle Data Pump 迁移",
    "ORACLE_ADG_DRILL": "Oracle Data Guard 演练",
    # 实施文档
    "READY": "已就绪",
    "BLOCKED_BY_REQUIRED_INPUTS": "等待必填参数",
    "BLOCKED_BY_REQUIRED_FACTS": "等待必要事实",
    "PARTIAL_EVIDENCE": "证据不完整",
    "REQUIRED": "必须实施",
    "ALREADY_SATISFIED": "当前已满足",
    "CONDITIONAL": "按条件实施",
    "BLOCKED": "已阻断",
    "USER_SUPPLIED": "用户提供",
    "VERIFIED": "已验证",
    "DERIVED": "系统推导",
    "TARGET_FACT": "运维目标已登记事实",
    "DEPLOYMENT_TOPOLOGY": "部署拓扑",
    "HOST_COLLECTOR": "主机采集数据",
    "POLICY_TEMPLATE": "策略模板",
    "EXPLICIT_USER_DECISION": "用户明确决定",
    "SQLPLUS": "SQL*Plus",
    "PSQL": "PostgreSQL psql",
    "RMAN": "RMAN",
    "DGMGRL": "Data Guard Broker 命令行",
    "SHELL": "Shell 命令",
    "CONFIG": "配置文件",
    "MANUAL": "人工操作",
    "BASH": "Bash",
    "CRSCTL": "CRSCTL",
    "SRVCTL": "SRVCTL",
    "ASMCMD": "ASMCMD",
    "DBCA": "DBCA",
    "OPATCH": "OPatch",
    "DATAPUMP": "Data Pump",
    # 正式报告
    "EXECUTIVE_SUMMARY": "执行摘要",
    "SCOPE": "报告范围",
    "ALERT_TIMELINE": "告警时间线",
    "INSPECTION_COVERAGE": "巡检覆盖情况",
    "RISK_OVERVIEW": "风险概览",
    "TREND": "趋势分析",
    "FINDINGS": "核验发现",
    "ROOT_CAUSE": "根因分析",
    "RECOMMENDATIONS": "处置建议",
    "ACTIONS": "已执行动作",
    "EVIDENCE_BOUNDARY": "证据边界",
    "EVIDENCE_APPENDIX": "证据附录",
    "GENERATING": "生成中",
    "RESOLVED": "已解决",
    "IMPROVED": "已改善",
    "UNCHANGED": "未改善",
    "DEGRADED": "已恶化",
    # DBA 工作项
    "INCIDENT_RESPONSE": "故障响应",
    "RISK_REMEDIATION": "风险整改",
    "PROBLEM_INVESTIGATION": "问题调查",
    "OPTIMIZATION": "优化事项",
    "OBSERVABILITY_GAP": "可观测性缺口",
    "RECOVERY_OBSERVATION": "恢复观察",
    "RECOVERY_DRILL": "恢复演练",
    "RECOVERY_PROFILE": "恢复策略",
    "P1": "一级紧急",
    "P2": "二级高优先",
    "P3": "三级普通",
    "P4": "四级低优先",
    "PENDING_TRIAGE": "等待分诊",
    "OPEN": "待处理",
    "IN_PROGRESS": "处理中",
    "WAITING": "等待中",
    "PENDING_VERIFICATION": "等待验证",
    "CLOSED": "已关闭",
    "TRIAGE": "分诊",
    "DIAGNOSIS": "诊断",
    "REMEDIATION": "处置",
    "APPROVAL": "审批",
    "EXECUTION": "执行",
    "FIXED": "已修复",
    "MITIGATED": "已缓解",
    "OBSERVED_STABLE": "观察期内保持稳定",
    "FALSE_POSITIVE": "误报",
    "ACCEPTED_RISK": "已接受风险",
    "DUPLICATE": "重复事项",
    "NO_ACTION": "无需处置",
    "CANNOT_REPRODUCE": "无法复现",
    "PROPOSAL": "处置建议",
    "ROUTING_RULE": "路由规则",
    "TARGET_RESPONSIBILITY": "Target 责任归属",
    "CLAIM": "主动领取",
    "INACTIVE": "停用",
    "LEAD": "组长",
    "MEMBER": "成员",
    "ACTION_REQUIRED": "需要处置",
    "MANUAL_INVESTIGATION": "需要人工调查",
    "NO_WORK_ITEM": "无需创建工作项",
    # 数据库工作负载报告
    "ORACLE_NATIVE": "Oracle 原生报告",
    "AIOPS_GENERATED": "AIOps 生成报告",
    "EXTERNAL_IMPORTED": "外部导入报告",
    "MYSQL_WORKLOAD": "MySQL 工作负载报告",
    "ORACLE_WORKLOAD": "Oracle 周期快照工作负载报告（非 AWR）",
    "ORACLE_WORKLOAD_DIFF": "Oracle 周期快照工作负载对比报告（非 AWR）",
    "MYSQL_WORKLOAD_DIFF": "MySQL 工作负载对比报告",
    "MYSQL_ACTIVITY": "MySQL 活动报告",
    "MYSQL_SQL_HEALTHCHECK": "MySQL SQL 健康检查",
    "POSTGRESQL_WORKLOAD": "PostgreSQL 工作负载报告",
    "POSTGRESQL_WORKLOAD_DIFF": "PostgreSQL 工作负载对比报告",
    "POSTGRESQL_ACTIVITY": "PostgreSQL 活动报告",
    "POSTGRESQL_SQL_HEALTHCHECK": "PostgreSQL SQL 健康检查",
    "POSTGRESQL_PGBADGER": "PostgreSQL pgBadger 报告",
    "MYSQL_DIGEST": "MySQL 语句摘要",
    "ORACLE_SQL_ID": "Oracle SQL 标识",
    "UNAVAILABLE": "不可用",
    "POSTGRESQL_QUERY_ID": "PostgreSQL 查询标识",
    "AVAILABLE": "可用",
    "DENIED": "权限不足",
    "TIMEOUT": "超时",
    "RESET": "统计已重置",
    "EVICTED": "数据已淘汰",
    "FIRING": "告警持续触发",
    "INFORMATIONAL": "信息事件",
    "WARNING": "警告",
    "PERMANENT": "永久表空间",
    "TEMPORARY": "临时表空间",
    "UNDO": "撤销表空间",
    # 证据缺口与诊断链路错误
    "MISSING_ASH": "缺少 ASH 历史会话数据",
    "MISSING_PRIMARY_RUN": "缺少主诊断运行记录",
    "MISSING_FINAL_RESULT": "缺少最终诊断结果",
    "UNREPORTABLE_FINAL_RESULT": "最终结果暂不支持生成报告",
    "MISSING_FINAL_ARTIFACT": "缺少最终报告产物",
    "USER_RESULT_UNAVAILABLE": "用户提供的结果不可用",
    "EVIDENCE_FACT_LIMIT_REACHED": "已达到证据事实数量上限",
    "VERIFICATION_EVIDENCE_MISSING": "缺少处置验证证据",
    "ACTION_VERIFIER_UNAVAILABLE": "动作验证器不可用",
    "EVIDENCE_NOT_COMPARABLE": "处理前后证据不可比较",
    "VERIFICATION_STATE_UNSUPPORTED": "当前验证状态不受支持",
    "SOURCE_AUTH_FAILED": "诊断源认证失败",
    "SOURCE_UNREACHABLE": "诊断源不可达",
    "DIAGNOSTIC_POLICY_DENIED": "诊断策略不允许数据库直连",
    "DB_DIRECT_NOT_CONFIGURED": "未配置数据库直连",
    "TARGET_INACTIVE": "运维目标未启用",
    "DIAGNOSTIC_SECRET_MISSING": "缺少诊断凭据",
    "TARGET_ENDPOINT_MISSING": "缺少运维目标连接地址",
    "TARGET_CONNECTIVITY_UNAVAILABLE": "运维目标当前不可连接",
    "VERSION_UNSUPPORTED": "数据库版本不受支持",
    "INSPECTION_OBSERVATION_MISSING": "缺少巡检观测结果",
    "INSPECTION_TEMPLATE_STEPS_MISSING": "巡检模板缺少检查步骤",
    "METRIC_SOURCE_UNAVAILABLE": "指标源不可用",
    "ACTION_TEMPLATE_UNAVAILABLE": "动作模板不可用",
    "EXECUTION_SECRET_MISSING": "缺少执行凭据",
    "EXECUTION_UNAVAILABLE_ADVISORY_PROVIDED": "无法自动执行，已提供人工建议",
    "MUTATION_EXECUTION_UNAVAILABLE": "受控变更执行不可用",
    "OPERATIONS_KNOWLEDGE_UNAVAILABLE": "运维知识不可用",
    "OPS_DELEGATION_RESULT_NOT_READY": "委派结果尚未就绪",
    "POLICY_MISSING": "缺少执行策略",
    "SECRET_UNAVAILABLE": "凭据不可用",
    "VERIFIED_ACTION_PARAMETERS_UNAVAILABLE": "缺少已验证的动作参数",
    "EXTERNAL_SOURCE_NOT_SOURCE_VERIFIED": "外部数据源未通过来源验证",
    "DISCONTINUITY": "数据采样不连续",
    "METRIC_COUNTER_RESET": "指标计数器已重置",
    "SAMPLING_BOUNDARY": "采样边界限制",
    "RPO_NOT_CONFIGURED": "未配置业务 RPO",
    "RTO_NOT_CONFIGURED": "未配置业务 RTO",
    "RESTORE_NOT_DEMONSTRATED": "恢复能力尚未实证",
    "LATEST_DRILL_FAILED": "最近一次恢复演练失败",
    "DRILL_STALE": "最近合格成功演练已超过要求周期",
    "POLICY_CHANGED_SINCE_DRILL": "演练后恢复策略已变更",
    "RPO_MEASUREMENT_MISSING": "合格演练缺少实测 RPO",
    "RTO_MEASUREMENT_MISSING": "合格演练缺少实测 RTO",
    "RPO_TARGET_BREACHED": "实测 RPO 超出业务目标",
    "RTO_TARGET_BREACHED": "实测 RTO 超出业务目标",
    "BACKUP_SOURCE_NOT_VERIFIED": "必需备份来源尚未验证",
    "VERIFICATION_ADVERSE": "验证发现不利变化",
    "ACTION_EFFECT_VERIFIED": "已验证动作达到预期效果",
    "EXPECTED_DIRECT_EFFECT_NOT_OBSERVED": "未观测到预期直接效果",
    "TARGET_ABSENT": "目标对象已消失",
    "BLOCKING_ABSENT": "阻塞关系已消失",
    "AUTH_FAILED": "数据库认证失败",
    "PRIVILEGE_MISSING": "数据库只读账号缺少所需权限",
    "PRIVILEGE_PROBE_CACHED_DENIAL": "近期权限拒绝的短期缓存，未重复查库",
    "OUTPUT_COLUMNS_MISMATCH": "查询结果列与诊断合同不一致",
    "OUTPUT_COLUMN_LIMIT_EXCEEDED": "查询结果列数超过安全上限",
    "OUTPUT_VALUE_TYPE_UNSUPPORTED": "查询结果包含不支持的数据类型",
    "OUTPUT_COLUMN_TYPE_MISMATCH": "查询结果列类型与诊断合同不一致",
    "OUTPUT_SCHEMA_INVALID": "查询结果结构校验失败",
    "RESULT_LIMIT_EXCEEDED": "查询结果超过安全行数上限",
    "QUERY_INCOMPATIBLE": "诊断查询与当前数据库版本不兼容",
    "QUERY_OBJECT_UNAVAILABLE": "诊断查询所需对象不可用",
    "QUERY_COLUMN_INVALID": "诊断查询使用了无效列",
    "QUERY_COLUMN_AMBIGUOUS": "诊断查询存在同名列歧义",
    "QUERY_SYNTAX_INVALID": "诊断查询语法无效",
    "QUERY_VALUE_FORMAT_INVALID": "诊断查询返回值格式无效",
    "INVESTIGATION_EVIDENCE_GAP": "仍有诊断证据缺口",
    "FINDING_COLUMN_MISSING": "发现卡片所需证据列缺失",
    "FINDING_VALUE_UNPARSEABLE": "发现卡片证据值无法解析",
    "FINDING_RATIO_UNDEFINED": "发现卡片比率无法计算",
    "SOURCE_NO_DATA": "监控源未返回有效数据",
    "ACTIVE_DIAGNOSIS": "主动诊断",
    "KNOWLEDGE_QA": "运维知识问答",
    "REPORT_CASE_QUERY": "历史诊断案例查询",
    "INSPECTION_QUERY": "巡检查询",
    "ALERT_QUERY": "告警查询",
    "MONITORING_QUERY": "当前监控查询",
    "RESOURCE_CATALOG_QUERY": "业务系统资源查询",
    "CONTROLLED_CHANGE": "受控变更",
}

AIOPS_DISPLAY_NAMES = MappingProxyType(_DISPLAY_NAMES)


def _load_metric_display_names() -> dict[str, str]:
    """从 Metric Catalog 读取全部指标展示名，避免另建易漂移的硬编码清单。"""
    catalog_path = (
        Path(__file__).resolve().parents[1]
        / "resources"
        / "metrics"
        / "baseline.v1.json"
    )
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    return {
        str(item["metric_code"]): str(item["name"])
        for item in payload.get("metrics", ())
    }


AIOPS_METRIC_DISPLAY_NAMES = MappingProxyType(_load_metric_display_names())

_DIMENSION_DISPLAY_NAMES = {
    "application_name": "应用名称",
    "cluster": "集群",
    "command": "命令类型",
    "container": "容器",
    "database": "数据库",
    "database_name": "数据库",
    "datname": "数据库",
    "device": "设备",
    "fstype": "文件系统类型",
    "host": "主机",
    "mountpoint": "挂载点",
    "namespace": "命名空间",
    "pod": "Pod",
    "role": "角色",
    "schema": "Schema",
    "schema_name": "Schema",
    "slot_name": "复制槽",
    "tablespace": "表空间",
    "tablespace_name": "表空间",
    "type": "表空间类型",
}
_HIDDEN_REPORT_DIMENSIONS = {
    "binding_id", "instance", "job", "source_id", "target_key",
}
_METRIC_FIELD_DISPLAY_NAMES = {
    "average": "平均值",
    "change": "期间变化量",
    "change_per_day": "日均变化量",
    "coverage_ratio": "采样覆盖率",
    "estimated_days_to_limit": "预计达到容量上限的剩余天数",
    "first": "期初值",
    "forecast_confidence": "预测置信度",
    "forecast_horizon_days": "预测周期（天）",
    "forecast_utilization_percent": "预测使用率",
    "forecast_value": "预测值",
    "history_elapsed_days": "历史覆盖天数",
    "last": "最新值",
    "latest": "当前值",
    "maximum": "最大值",
    "minimum": "最小值",
    "sample_count": "有效样本数",
    "sample_p95": "采样值 P95（非执行延迟 P95）",
    "trend_slope_per_day": "每日趋势斜率",
}
_FIELD_TOKEN_DISPLAY_NAMES = {
    "active": "活动", "age": "年龄", "allocated": "已分配", "analyze": "分析",
    "applier": "应用线程", "archive": "归档", "archived": "已归档", "at": "时间",
    "autoanalyze": "自动分析", "autovacuum": "自动清理", "average": "平均",
    "backend": "后端进程", "backends": "后端进程数", "bin": "二进制",
    "binlog": "二进制日志", "blks": "数据块", "blocking": "阻塞方",
    "blocks": "块数", "buffers": "缓冲区", "bytes": "字节", "cardinality": "基数",
    "catalog": "系统目录", "chain": "链路", "channel": "通道",
    "checkpoint": "检查点", "checkpoints": "检查点", "checksum": "校验和",
    "class": "类别", "command": "命令", "commit": "提交", "configured": "已配置",
    "context": "上下文", "count": "数量", "current": "当前", "database": "数据库",
    "dead": "无效", "deadlocks": "死锁数", "default": "默认", "depth": "深度",
    "dest": "目标位置", "device": "设备", "digest": "语句摘要", "disk": "磁盘",
    "elapsed": "耗时", "end": "结束", "error": "错误", "estimated": "估算",
    "event": "事件", "examined": "扫描", "execution": "执行", "executions": "执行次数",
    "expire": "过期", "expired": "已过期", "failed": "失败", "failures": "失败数",
    "fg": "前台", "file": "文件", "files": "文件数", "flashback": "闪回",
    "force": "强制", "foreground": "前台", "format": "格式", "fpi": "全页镜像",
    "fra": "快速恢复区", "free": "可用", "full": "已满", "gtid": "GTID",
    "headroom": "余量", "hit": "命中", "host": "主机", "id": "标识",
    "idle": "空闲", "index": "索引", "indexes": "索引数", "input": "输入",
    "instance": "实例", "io": "I/O", "is": "是否", "job": "任务", "lag": "延迟",
    "last": "最近", "latency": "延迟", "limit": "上限", "live": "有效",
    "locked": "锁定", "log": "日志", "logging": "日志记录", "logs": "日志",
    "max": "最大", "maximum": "最大", "mb": "MB", "member": "成员",
    "metric": "指标", "mod": "修改", "mode": "模式", "modifiable": "可动态修改",
    "modification": "修改", "modified": "已修改", "ms": "毫秒", "multipasses": "多次通过",
    "multixact": "多事务", "n": "数量", "name": "名称", "number": "编号",
    "object": "对象", "on": "启用", "onepass": "单次通过", "open": "打开",
    "output": "输出", "owner": "属主", "parameter": "参数", "per": "每",
    "percent": "百分比", "picoseconds": "皮秒", "port": "端口", "processed": "已处理",
    "query": "查询", "questions": "查询数", "read": "读取", "reads": "读取次数",
    "receiver": "接收线程", "reclaimable": "可回收", "records": "记录数",
    "recovery": "恢复", "requested": "请求触发", "reset": "重置", "resource": "资源",
    "retained": "保留", "role": "角色", "rollback": "回滚", "rows": "行数",
    "running": "运行中", "runtime": "运行时", "schema": "Schema", "second": "秒",
    "seconds": "秒", "sent": "返回", "service": "服务", "session": "会话",
    "since": "自", "slot": "复制槽", "source": "来源", "space": "空间",
    "sql": "SQL", "start": "开始", "started": "开始", "state": "状态",
    "stats": "统计信息", "status": "状态", "storage": "存储", "successful": "成功",
    "sync": "同步", "table": "表", "tables": "表数", "tablespace": "表空间",
    "temp": "临时空间", "threads": "线程数", "time": "时间", "timed": "定时触发",
    "total": "总计", "transaction": "事务", "tup": "元组", "tuple": "元组",
    "type": "类型", "undo": "撤销", "unexpired": "未过期", "unit": "单位",
    "update": "更新时间", "uptime": "运行时长", "used": "已使用", "username": "用户名",
    "utilization": "使用率", "vacuum": "清理", "value": "值", "version": "版本",
    "wait": "等待", "waited": "等待耗时", "waiting": "等待方", "waits": "等待次数",
    "wal": "WAL", "window": "时间窗口", "without": "缺少", "write": "写入",
    "writes": "写入次数", "written": "已写入", "xact": "事务", "xid": "事务 ID",
    "xmin": "xmin",
}

_METRIC_WITH_DIMENSIONS_PATTERN = re.compile(
    rf"(?P<code>{'|'.join(re.escape(code) for code in sorted(AIOPS_METRIC_DISPLAY_NAMES, key=len, reverse=True))})"
    r"\[(?P<dimensions>[^\]\n]*)\]"
)
_METRIC_CODE_PATTERN = re.compile(
    rf"(?<![A-Za-z0-9_.-])({'|'.join(re.escape(code) for code in sorted(AIOPS_METRIC_DISPLAY_NAMES, key=len, reverse=True))})(?![A-Za-z0-9_.-])"
)
_METRIC_FIELD_PATTERN = re.compile(
    rf"(?<![A-Za-z0-9_])({'|'.join(re.escape(code) for code in sorted(_METRIC_FIELD_DISPLAY_NAMES, key=len, reverse=True))})\s*="
)
_FIELD_ASSIGNMENT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])([a-z][a-z0-9_]{1,})\s*="
)
_DISPLAY_CODE_PATTERN = re.compile(r"(?<![A-Za-z0-9_])([A-Z][A-Z0-9_]{2,})(?![A-Za-z0-9_])")
_MARKDOWN_CODE_PATTERN = re.compile(
    r"(```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]*`)"
)


def display_name(value: object, *, default: str | None = None) -> str:
    """返回稳定代码的中文展示名，未知值按调用方要求回退。"""
    text = str(value)
    return AIOPS_DISPLAY_NAMES.get(text, default if default is not None else text)


def display_metric_name(metric_code: object) -> str:
    """返回 Metric Catalog 中的中文名称，未知代码不泄露到正式报告。"""
    return AIOPS_METRIC_DISPLAY_NAMES.get(str(metric_code), "监控指标")


def display_field_name(field_name: object) -> str:
    """把证据字段名转换为面向 DBA 的展示标签。"""
    text = str(field_name)
    if text in _METRIC_FIELD_DISPLAY_NAMES:
        return _METRIC_FIELD_DISPLAY_NAMES[text]
    if text in _DIMENSION_DISPLAY_NAMES:
        return _DIMENSION_DISPLAY_NAMES[text]
    tokens = text.lower().split("_")
    translated = [_FIELD_TOKEN_DISPLAY_NAMES.get(token) for token in tokens]
    if all(translated):
        return "".join(translated)
    return "观测字段"


def _display_dimension_value(value: object) -> str:
    text = str(value)
    return AIOPS_DISPLAY_NAMES.get(text.upper(), text)


def display_metric_dimensions(value: object) -> str:
    """隐藏采集器内部 Label，只保留能帮助 DBA 理解对象的业务维度。"""
    if isinstance(value, dict):
        pairs = [(str(key), item) for key, item in value.items()]
    else:
        text = str(value or "").strip().strip("[]{}")
        if not text or text == "-":
            return ""
        pairs = []
        for part in text.split(","):
            key, separator, item = part.strip().partition("=")
            if separator:
                pairs.append((key.strip(), item.strip()))
    visible = [
        f"{_DIMENSION_DISPLAY_NAMES.get(key, display_field_name(key))}："
        f"{_display_dimension_value(item)}"
        for key, item in pairs
        if key not in _HIDDEN_REPORT_DIMENSIONS and not key.startswith("__")
    ]
    return f"（{'，'.join(visible)}）" if visible else ""


def format_metric_value(value: object, unit: object = "") -> str:
    """按 Metric Catalog 单位生成可直接进入正式报告的值。"""
    normalized_unit = str(unit or "")
    if value is None or value == "":
        return "无有效值"
    if normalized_unit == "state":
        return _display_dimension_value(value)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return str(value)
    numeric = float(value)
    if normalized_unit in {"bytes", "bytes_per_second"}:
        suffix = "/秒" if normalized_unit == "bytes_per_second" else ""
        units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
        index = 0
        magnitude = abs(numeric)
        while magnitude >= 1024 and index < len(units) - 1:
            numeric /= 1024
            magnitude /= 1024
            index += 1
        return f"{numeric:,.2f} {units[index]}{suffix}"
    rendered = f"{numeric:,.2f}".rstrip("0").rstrip(".")
    suffixes = {
        "count": "",
        "errors_per_second": " 次错误/秒",
        "milliseconds": " 毫秒",
        "percent": "%",
        "queries_per_second": " 次查询/秒",
        "transactions_per_second": " 个事务/秒",
        "waits_per_second": " 次等待/秒",
    }
    return f"{rendered}{suffixes.get(normalized_unit, f' {normalized_unit}' if normalized_unit else '')}"


def display_field_value(field_name: object, value: object) -> str:
    """为结构化关键观测补充单位并翻译枚举值。"""
    field = str(field_name)
    if value is None or value == "":
        return "—"
    if field.endswith("_bytes"):
        return format_metric_value(value, "bytes")
    if field.endswith("_percent"):
        return format_metric_value(value, "percent")
    if field.endswith("_seconds"):
        return f"{format_metric_value(value)} 秒"
    if field.endswith("_ms"):
        return format_metric_value(value, "milliseconds")
    if field.endswith("_mb"):
        return f"{format_metric_value(value)} MB"
    if isinstance(value, str):
        return _display_dimension_value(value)
    return str(value)


def display_text(value: object) -> str:
    """翻译自然语言中的已登记稳定代码，不改写 SQL 与普通技术名词。"""
    parts = _MARKDOWN_CODE_PATTERN.split(str(value))

    def translate(part: str) -> str:
        part = _METRIC_WITH_DIMENSIONS_PATTERN.sub(
            lambda match: (
                f"{display_metric_name(match.group('code'))}"
                f"{display_metric_dimensions(match.group('dimensions'))}"
            ),
            part,
        )
        part = _METRIC_CODE_PATTERN.sub(
            lambda match: display_metric_name(match.group(1)), part
        )
        part = _METRIC_FIELD_PATTERN.sub(
            lambda match: f"{_METRIC_FIELD_DISPLAY_NAMES[match.group(1)]}：", part
        )
        part = _FIELD_ASSIGNMENT_PATTERN.sub(
            lambda match: f"{display_field_name(match.group(1))}：", part
        )
        return _DISPLAY_CODE_PATTERN.sub(
            lambda match: AIOPS_DISPLAY_NAMES.get(
                match.group(1), match.group(1)
            ),
            part,
        )

    return "".join(
        part
        if index % 2
        else translate(part)
        for index, part in enumerate(parts)
    )


def display_payload(value: Any) -> Any:
    """递归为模型输入翻译稳定代码，同时保持容器结构不变。"""
    if isinstance(value, dict):
        return {key: display_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [display_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(display_payload(item) for item in value)
    if isinstance(value, str):
        return display_text(value)
    return value


def dictionary_for_payload(value: Any) -> dict[str, str]:
    """只返回当前模型输入实际包含的展示字典，避免无关字典污染上下文。"""
    codes: set[str] = set()

    def collect(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                collect(key)
                collect(child)
            return
        if isinstance(item, (list, tuple, set)):
            for child in item:
                collect(child)
            return
        if isinstance(item, str):
            codes.update(_DISPLAY_CODE_PATTERN.findall(item))

    collect(value)
    return {
        code: AIOPS_DISPLAY_NAMES[code]
        for code in sorted(codes)
        if code in AIOPS_DISPLAY_NAMES
    }
