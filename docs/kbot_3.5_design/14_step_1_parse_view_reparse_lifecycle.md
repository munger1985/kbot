# 步骤 1 详细设计：Parse View 与重解析生命周期

## 职责

`KBOT_KC_PARSE_VIEW` 保存一个 Document Version 在指定解析策略下的结构化解析结果。它是派生数据，不是历史档案：重解析成功后必须替换并删除旧 View、Evidence 和相关检索投影；原始 Document Version 不删除。

Document Version 可以同时拥有不同用途的 View Type（例如 TEXT、VISUAL、SPREADSHEET）。但对同一 `document_version_id + view_type + coverage_key`，只允许一个 ACTIVE View，且不长期保留旧 Active 历史。

## 表字段

| 字段 | 类型/规则 | 说明 |
| --- | --- | --- |
| `parse_view_id` | `NUMBER(38)` PK identity | 解析产物标识 |
| `collection_id`, `bundle_id`, `document_id`, `document_version_id` | 非空 `NUMBER(38)` | 父链与 Scope 加速关联 |
| `view_type` | `VARCHAR2(24)` 非空 | `TEXT/VISUAL/HYBRID/SPREADSHEET` |
| `coverage_key` | `VARCHAR2(128)` 非空 | 解析覆盖范围；全文件为 `FULL`，局部页段使用稳定范围键 |
| `view_status` | `VARCHAR2(16)` 非空 | `BUILDING/ACTIVE/FAILED/DELETING` |
| `parse_policy_snapshot_json` | JSON CLOB 非空 | 引擎、参数、模型、提示词与分块策略快照 |
| `parser_version` | `VARCHAR2(128)` 非空 | Worker/解析器发布版本 |
| `quality_score` | `NUMBER(8,6)` 可空 | 质量评估归一化分数 |
| `quality_report_json` | JSON CLOB 可空 | 质量门、覆盖范围、告警与统计 |
| `output_uri` | `VARCHAR2(2048)` 可空 | 大型中间产物或结构化输出 URI |
| `started_at`, `completed_at` | 带时区时间戳 | 处理时序 |
| `failure_code`, `failure_message` | 可空 | 失败摘要 |
| 审计列 | 基础约定 | 创建/处理服务审计 |

索引：`(document_version_id, view_type, coverage_key, view_status)` 用于替换检查；`(collection_id, view_status)` 用于巡检。ACTIVE 唯一性由 KC 切换事务保证；不依赖长期保留候选历史。

## 成功后替换

```text
已有 ACTIVE View + Evidence
  → 新 Parse View BUILDING
  → 新 Evidence / Discovery 投影写入不可见暂存状态
  → 质量门通过
  → 同一事务：新 View/Evidence ACTIVE，物理删除旧 View/Evidence
  → 事务提交后删除旧不可变解析工件；失败时由存储巡检清除孤儿
```

重解析期间，查询只读取旧 ACTIVE View 和对应 ACTIVE Evidence，因此不会出现检索空窗。切换事务保证同一范围不会存在两个 ACTIVE View；工件文件清理失败不影响数据库可见性，但需由存储巡检清除孤儿。

新候选解析最终失败时：同一事务删除候选 View/Evidence，提交后删除 Worker 已上传工件；旧 ACTIVE View 保持不变。若该 Document Version 从未有成功 View，则 Member 进入 `FAILED`，不存在可检索结果。可重试的 TRANSIENT 失败保留同一候选 View 和幂等 Evidence，等待租约重领。

## 触发条件与参数变更

- 解析失败后的显式重试：沿用或显式指定新的解析策略。
- Parser、OCR/VLM、Evidence 规划参数或质量门变更：创建新的 BUILDING View，不覆盖旧 View。
- Collection 的 Embedding 模型变化不创建 Parse View，而只重建该 Collection 的 Evidence/Discovery 向量；只有全局维度变化才需要全应用 DDL 与重建。
- 内容变化：创建新的 Document Version，再为新 Version 创建 Parse View；不称为重解析。
- 安全隔离/误解析需要立即撤回旧结果时，使用单独的紧急撤回操作，直接撤销旧 Evidence 可见性；它不走常规“成功后替换”流程。

常规重解析 API 不允许直接删除 Active View；必须通过候选构建、质量门和切换流程。管理页面只展示当前 Active 结果与正在执行的重解析状态，不把旧解析结果作为可恢复历史提供。
