# 步骤 1 详细设计：Discovery Object

## 定位

`KBOT_KC_DISCOVERY_OBJECT` 是某次 Bundle Revision 的可重建检索画像，用于第一阶段回答“哪一个业务对象或附件值得继续查”。它不是事实来源、不是 Citation 的最小单元，也不替代 Evidence。

```text
自然语言问题
  → Discovery Object：召回当前 Revision 的 Bundle / Document Member
  → Evidence：仅在命中的 Member 所指 Version 内检索并引用
```

它必须归属 `bundle_revision_id`：标题、Facet、文件角色和声明名称均可随 Revision 改变；不能把这些内容写入可跨 Revision 复用的 Evidence。

## 表字段

| 字段 | 类型/规则 | 说明 |
| --- | --- | --- |
| `discovery_object_id` | `NUMBER(38)` PK identity | 检索画像标识 |
| `collection_id`, `bundle_id`, `bundle_revision_id` | 非空 `NUMBER(38)` | Scope、业务对象与不可变来源快照 |
| `bundle_revision_document_id` | 可空 `NUMBER(38)` | `DOCUMENT` 对象必填，`BUNDLE` 对象为空 |
| `document_id`, `document_version_id` | 可空 `NUMBER(38)` | 附件对象的逻辑身份和当前快照使用的内容 Version |
| `object_type` | `VARCHAR2(16)` 非空 | `BUNDLE/DOCUMENT` |
| `profile_key` | `VARCHAR2(256)` 非空 | Revision 内稳定键：`bundle` 或 `member:{external_document_id}` |
| `display_title` | `VARCHAR2(512)` 非空 | 当前 Revision 的标题或声明文件名，供卡片展示 |
| `profile_text` | `CLOB` 非空 | 面向 Discovery 的完整可检索画像文本 |
| `facet_json` | JSON CLOB 可空 | Revision Facet 与 Document 可检索投影；不作为事实真源 |
| `coverage_json` | JSON CLOB 非空 | 已纳入的 Member、Version、Parse View、Evidence 数量和缺失说明 |
| `profile_hash` | `VARCHAR2(64)` 非空 | 规范化输入的 hash；相同输入不重复构建 |
| `profile_schema_version` | `VARCHAR2(32)` 非空 | 画像拼接/Facet 投影规则版本 |
| `embedding` | Oracle `VECTOR` 可空 | 基于 `profile_text` 的语义召回向量 |
| `embedding_model_key` | `VARCHAR2(128)` 可空 | 向量模型与配置标识 |
| `security_level` | `NUMBER(3)` 非空 | 从 Revision/Member/Version 的最严格有效等级派生 |
| `quality_score` | `NUMBER(8,6)` 可空 | 覆盖度和画像质量，不表示原文真实性 |
| `discovery_status` | `VARCHAR2(16)` 非空 | `STAGED/ACTIVE/DELETING/FAILED` |
| 审计列 | 基础约定 | 生成服务、时间与错误追踪 |

不存 `app_id/domain_id`；均由 `collection_id` 关联。`profile_text` 可以包含 Revision 标题、Facet、Member 角色/名称、实际 MIME、文档摘要、关键词、章节/表格概要和可追溯的覆盖摘要；这些内容用于找对象，不作为最终事实引用。

## 对象粒度与约束

- 每个 Bundle Revision 恰有一个 `BUNDLE` 画像：`UK(bundle_revision_id, object_type)`。
- 每个可见 Revision Document Member 至多一个 `DOCUMENT` 画像：`UK(bundle_revision_document_id, object_type)`。没有可用 Version 的失败附件可以不建画像；其缺失情况进入 Bundle 的 `coverage_json`。
- `profile_key` 用于 Worker 幂等和调试：`UK(bundle_revision_id, profile_key)`。
- 索引 `(collection_id, discovery_status, security_level)` 供权限预过滤；索引 `(bundle_revision_id, object_type, discovery_status)` 供当前 Revision 查询；Oracle Text 索引 `profile_text`、Vector 索引 `embedding`。
- 查询同时限制 Bundle 的 `current_revision_id=bundle_revision_id`、Revision 状态 `READY/PARTIAL` 和 `discovery_status=ACTIVE`。因此历史 Revision 即使清理任务尚未执行，也不会被召回。

## 构建、切换与失败

`PROFILE` Job 在 Manifest、Member 和其 Active Parse View/Evidence 就绪后构建画像；`INDEX` Job 再生成全文/向量索引。输入为当前 Revision 快照与可用解析产物，`profile_hash` 相同则幂等跳过。

新 Revision 的对象先写 `STAGED`。当 Revision 达到 `READY` 或允许切换的 `PARTIAL` 时，在同一切换流程中：更新 Bundle `current_revision_id`，激活该 Revision 的全部合格画像，并撤销旧 Revision 画像可见性。新画像失败不能破坏旧 Revision 的 Discovery；若首次 Revision 无可用画像，则 Revision 不能以“可检索”为由标记 READY。

`coverage_json` 必须显式记录未入库、解析失败或被隔离的 Member。Discovery 卡片可据此展示“部分附件不可用”，但 Evidence API 不得把失败原因当作回答内容。

## 与 Evidence 的边界

Discovery 命中 `BUNDLE` 时，KC 将其展开为该 Revision 的 READY Member；命中 `DOCUMENT` 时，只开放该 Member 指定的 `document_version_id`。之后 Evidence 查询仍以 ACTIVE Parse View、Evidence 状态和安全等级过滤。

`profile_text` 中的摘要、关键词或 VLM 归纳只能作为召回信号。最终回答必须引用 Evidence 的 `content + locator_json`；若 Discovery 命中而没有足够 Evidence，Skill 应说明“找到相关对象但没有可引用内容”，不得用画像文本填充事实答案。
