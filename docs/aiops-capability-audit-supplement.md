# AIOps 审计报告补充

> 补充日期: 2026-07-16
> 基于: aiops-capability-audit.md + NexusCube/kbot3 双项目代码交叉验证

---

## 一、审计报告未覆盖的发现

### 1. kbot3 独有: Slack 告警通道 (部分弥补感知缺口)

审计报告标注告警被动接入为 ❌，但 kbot3 实际有一个 Slack 集成通道：

| 文件 | 功能 |
|------|------|
| `api/controllers/slack_controller.py` | Slack slash command 接收 |
| `api/routers/slack_router.py` | Slack webhook 路由 |
| `slack_controller.py` | Slack 事件处理 |

**评估**: 这是一个受控的 Slack Bot 入口，不支持 Prometheus AlertManager 原生 webhook 格式。可用于人工通过 Slack 触发诊断，但不能自动接收监控告警。报告应修正为 ⚠️ 部分实现 (~20%)。

### 2. NexusCube 独有缺失项

| 缺失项 | kbot3 状态 | NexusCube 状态 | 影响 |
|--------|-----------|---------------|------|
| HITL 超时检测 CLI | `kbot_hitl_timeout_check.py` | ❌ 不存在 | NexusCube 挂起请求无人清理 |
| HITL DDL | `docs/database/kbot_db_change_ddl_3.3.1_hitl.sql` | ❌ 不存在 | NexusCube 缺少建表脚本 |
| HITL 设计文档 | `docs/ops_hitl_design.md` | ❌ 不存在 | 运维文档缺失 |
| Slack 集成 | 3 个文件 | ❌ 不存在 | 无即时通讯告警入口 |
| CLI 指标查询 | ❌ 不存在 | `cli/db_metric_cli.py` | kbot3 缺少命令行工具 |
| 过期备份文件 | ❌ 不存在 | `db_analysis_skill_core.py.bak` | 代码仓库污染 |

### 3. 审计完全遗漏的维度

#### 3a. 数据库差异对诊断工具的影响

| 引擎 | NexusCube | kbot3 |
|------|----------|-------|
| 主数据库 | PostgreSQL | Oracle 23ai |
| 全文检索 | ParadeDB (pg_bm25) | Oracle Text |
| 向量检索 | pgvector | Oracle AI Vector Search |
| 诊断工具适配 | PostgreSQL 专家 SQL | Oracle 专家 SQL |

`diagnostic_tools.py` 在两个项目中已有三引擎覆盖 (Oracle/PostgreSQL/MySQL)，但 **SQL 模板的测试覆盖**未被审计。

#### 3b. 多项目部署架构差异

| 维度 | NexusCube | kbot3 |
|------|----------|-------|
| 部署模式 | 9 个微服务独立进程 | 8 个微服务独立进程 |
| 前端 | NexusCubeUI (React) | 无 (纯后端) |
| 配置管理 | TOML (base + env) | TOML (base + env) |
| 内存/状态 | 无 | 无 (无 Redis) |

---

## 二、审计评分修正

| 阶段 | 原始评分 | 修正评分 | 修正原因 |
|------|---------|---------|---------|
| 感知 (Sense) | ~60% | ~65% | Slack 通道提供部分人工告警接入能力 |
| 诊断 (Diagnose) | ~80% | ~80% | 评分不变，两项目实现一致 |
| 决策 (Decide) | ~50% | ~50% | 评分不变 |
| 执行 (Execute) | ~75% | ~75% | 评分不变 |
| 验证 (Verify) | ~10% | ~10% | 评分不变，两项目同样缺失 |

### NexusCube 特有评分调整

| 维度 | 评分 | 说明 |
|------|------|------|
| 运维完整性 | ~85% | 无 HITL 超时清理、无 Slack |
| 代码整洁度 | ~90% | 存在 .bak 过期备份文件 |

---

## 三、补充后的完整偏差矩阵

| # | 缺口 | 严重度 | kbot3 | NexusCube | 审计覆盖 |
|---|------|--------|-------|-----------|---------|
| 1 | 验证闭环 (Verify) | 🔴 致命 | ❌ | ❌ | ✅ 已覆盖 |
| 2 | 回滚执行 | 🔴 致命 | ❌ | ❌ | ✅ 已覆盖 |
| 3 | 告警 Webhook 接入 | 🟠 高 | ⚠️ Slack | ❌ | ⚠️ 需修正 |
| 4 | 日志采集 (os_log) | 🟠 高 | ❌ | ❌ | ✅ 已覆盖 |
| 5 | 条件分支执行 | 🟠 高 | ❌ | ❌ | ✅ 已覆盖 |
| 6 | 风险动态评分 | 🟡 中 | ❌ | ❌ | ✅ 已覆盖 |
| 7 | 执行频次熔断 | 🟡 中 | ❌ | ❌ | ✅ 已覆盖 |
| 8 | 执行前校验 | 🟡 中 | ❌ | ❌ | ✅ 已覆盖 |
| 9 | 定时巡检 (Cron) | 🟡 中 | ❌ | ❌ | ✅ 已覆盖 |
| 10 | 并行步骤执行 | 🟢 低 | ❌ | ❌ | ✅ 已覆盖 |
| 11 | 跨实例关联分析 | 🟢 低 | ❌ | ❌ | ✅ 已覆盖 |
| 12 | HITL 超时清理 | 🟠 高 | ✅ | ❌ | ❌ **遗漏** |
| 13 | 即时通讯集成 | 🟡 中 | ✅ Slack | ❌ | ❌ **遗漏** |

---

## 四、改造优先级建议

按「ROI × 风险」排序：

```
P0 (立即): 验证闭环 + 回滚执行 — 当前无自愈成功/失败的判断标准
P1 (本周): 告警 Webhook + 日志采集 — 感知能力补齐
P2 (本月): 条件分支 + 风险评分 + 执行频次熔断 — 决策/执行安全加固
P3 (下月): 并行执行 + 跨实例分析 + Cron 定时巡检
```
