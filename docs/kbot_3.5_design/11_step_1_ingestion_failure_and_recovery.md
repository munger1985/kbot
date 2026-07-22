# 步骤 1 详细设计：入库失败、部分可用与恢复

## 基本原则

KC 区分“来源快照”与“处理结果”：Bundle Revision 的 Manifest、来源修订和声明的附件清单一经接收不可覆盖；解析、下载和索引状态可以重试并恢复。对当前已发布且不变更的 KM Asset，重点是可靠接收、局部失败可见与后续修复，不需要围绕来源更新建立复杂流程。

```text
有效请求
  → ACCEPTED（来源快照已持久化）
  → PROCESSING
  → READY / PARTIAL / FAILED
           │        │
           └─ 显式重试或补传 ──► PROCESSING ──► READY / PARTIAL / FAILED
```

`READY/PARTIAL/FAILED` 是当前处理结论，不是禁止恢复的不可变终态；不可变的是该 Revision 的来源 Manifest、声明附件和来源身份。来源内容真正变化时才创建新的 Bundle Revision。

## 上传阶段的失败边界

| 情况 | KC 行为 | Portal 行为 |
| --- | --- | --- |
| 鉴权、Domain/Collection、Manifest 格式、重复附件 ID、必需文件 part 缺失 | `4xx` 拒绝；不创建 Revision | 保持未处理并修正后重试 |
| 文件大小、MIME、hash、病毒/内容安全校验失败 | `4xx/422` 拒绝整个请求；清理临时对象 | 修正来源或标记失败后重试 |
| Portal 在上传前下载附件失败，且在 `document_failures` 显式声明 | 接收 Bundle Revision；创建“来源不可用”成员记录 | 可稍后补传该成员 |
| 对象存储发布或数据库写入失败 | 不返回 accepted；回滚数据库并清理/标记孤儿暂存对象 | 以同一请求安全重试 |

Portal 不能静默漏传：每个附件描述必须恰好对应一个文件 part，或恰好对应一个 `document_failures` 项。两者皆无是请求错误；两者同时存在也是请求错误。

## 附件成员状态

每个 Bundle Revision 的附件清单由 `KBOT_KC_BUNDLE_REVISION_DOCUMENT` 表保存，至少包含 `bundle_revision_id`、`collection_id`、`document_id`、可空 `document_version_id`、`role`、顺序、`required_flag`、`member_status` 与失败摘要。

`member_status`：`DECLARED`、`RECEIVED`、`SOURCE_UNAVAILABLE`、`PARSING`、`READY`、`FAILED`、`CANCELLED`。`SOURCE_UNAVAILABLE` 表示 Portal 已知但未能下载的来源附件；它不是 Parser 错误。补传成功时在同一来源快照下创建/关联 Document Version，并将成员推进到 `RECEIVED`，再投递解析任务。

## Parser 失败与 Bundle 状态归约

Parser 的短暂错误只更新 Job 的尝试次数、租约和下次重试时间，不改变 Bundle Revision 的来源数据。达到有限重试上限、判定为不支持格式或内容不可解析时，将相应成员标记为 `FAILED` 并保留结构化错误码。

所有成员到达稳定状态后，按以下规则计算 Revision 和 Bundle 的当前可用性：

- `READY`：Manifest 已生成可检索内容，且所有已接收/必需成员均有可用 Evidence；无声明失败或解析失败成员。
- `PARTIAL`：Manifest 有可检索内容，且至少一个成员或 Manifest 本身有可用 Evidence，但存在 `SOURCE_UNAVAILABLE`、`FAILED` 或仍待人工补传的成员。
- `FAILED`：Manifest 处理失败，或没有任何可检索 Evidence。

普通检索只使用 READY 成员的 Evidence。Discovery 可以显示 Bundle 为 `PARTIAL` 及附件状态，但不向普通用户暴露内部错误详情；管理 API 可查看受权限保护的失败原因。

## 补传与重试

- **解析重试**：通过受限内部/管理接口重新投递失败 Document Version 的 Job；不创建新 Bundle Revision。
- **附件补传**：通过受限的 KM Asset 补传接口，将文件绑定到该 Revision 中 `SOURCE_UNAVAILABLE` 的外部文档 ID；校验其名称、MIME、来源 URL 与声明一致，再创建 Document Version。
- **Revision 重算**：每次成员状态或 Evidence 可用性变化后，KC 在事务中重算 Revision 状态；如果它是 Bundle 的 `current_revision_id`，同步更新 Bundle `availability_status`。
- **来源内容变更**：只允许通过新的 Bundle Revision 处理，不能用补传接口替换已接收 Document Version 的内容。

补传与重试必须记录操作者、原因、原错误和新 Job/Document Version ID。多次相同补传以 `external_document_id + content_hash` 幂等；不同 hash 视为来源变更，拒绝在当前 Revision 内覆盖。
