# 步骤 1 详细设计：Bundle 与 Bundle Revision

## 为什么需要两张表

Bundle 是一个来源业务对象的稳定身份，例如一个 Metadb Asset；它不是一次上传。来源对象的标题、Facet、Manifest 或附件清单变化时，不能覆盖正在被检索的旧内容。因此采用：

```text
Collection
  └─ Bundle（稳定来源身份）
       └─ Bundle Revision（不可变的来源快照）
            └─ Document / Document Version / Parse View / Evidence
```

`KBOT_KC_BUNDLE` 只保存来源唯一键、当前可用 Revision 指针和聚合状态。`KBOT_KC_BUNDLE_REVISION` 保存每次来源修订的 Manifest、附件清单指纹、状态和审计。新 Revision 解析期间，Discovery/Evidence 继续使用 Bundle 的 `current_revision_id`；只有新 Revision 完成索引后才原子切换，避免检索空窗。

## `KBOT_KC_BUNDLE`

| 字段 | 类型/规则 | 说明 |
| --- | --- | --- |
| `bundle_id` | `NUMBER(38)` PK identity | 稳定内部身份 |
| `collection_id` | 非空 `NUMBER(38)` | 唯一 Scope 来源 |
| `source_system` | `VARCHAR2(64)` 非空 | 首期 `metadb`，由受信任 Adapter 注入 |
| `source_type` | `VARCHAR2(64)` 非空 | 首期 `KM_ASSET`，由 Adapter 注入 |
| `source_id` | `VARCHAR2(256)` 非空 | 来源业务主键，如 Asset ID |
| `current_revision_id` | `NUMBER(38)` 可空 | 最后一个可检索 Bundle Revision |
| `availability_status` | `VARCHAR2(16)` 非空 | `EMPTY/READY/PARTIAL/FAILED/DELETING` |
| `row_version` + 审计列 | 基础约定 | 管理面并发与审计 |

约束与索引：`UK(collection_id, source_system, source_type, source_id)`；索引 `(collection_id, availability_status)` 供 Discovery 过滤和 APEX 列表。Bundle 不保存可变标题、Facet、来源修订或附件清单；这些属于 Revision，APEX 视图通过 `current_revision_id` 展示当前值。

## `KBOT_KC_BUNDLE_REVISION`

| 字段 | 类型/规则 | 说明 |
| --- | --- | --- |
| `bundle_revision_id` | `NUMBER(38)` PK identity | 不可变快照身份 |
| `collection_id`, `bundle_id` | 非空 `NUMBER(38)` | Scope 与父对象关联 |
| `revision_no` | 非空 `NUMBER(19)` | Bundle 内单调递增版本号 |
| `source_revision` | `VARCHAR2(256)` 非空 | 上游修订号；KM Asset 首期为 `last_update_time` |
| `snapshot_fingerprint` | `VARCHAR2(64)` 非空 | 规范化 Manifest 与声明附件描述的 SHA-256；实际文件 hash 属于 Document Version |
| `manifest_json` | JSON CLOB 非空 | 接收时的完整规范化来源主信息 |
| `title` | `VARCHAR2(512)` 非空 | 当前来源标题的该次快照 |
| `canonical_url` | `VARCHAR2(2048)` 可空 | 来源页面链接 |
| `security_level` | `NUMBER(3)` 非空 | 该 Revision 的有效安全等级 |
| `facet_json` | JSON CLOB 可空 | 用于检索/渲染的来源 Facet |
| `status` | `VARCHAR2(16)` 非空 | 见下方生命周期 |
| `accepted_at`, `completed_at` | 带时区时间戳 | 接收和最终完成时间 |
| `failure_code`, `failure_message` | 可空 | 仅最终失败原因，不替代 Job 明细 |
| 审计列 | 基础约定 | 接收者与处理者审计 |

约束与索引：`UK(bundle_id, revision_no)`、`UK(bundle_id, source_revision)`；索引 `(bundle_id, status)` 与 `(collection_id, status, accepted_at)`。同一 `source_revision` 不允许产生两份不同快照。

## 修订幂等与切换规则

1. Adapter 按 `collection_id + source_system + source_type + source_id` 定位或创建 Bundle。
2. 同一 `source_revision` 且 `snapshot_fingerprint` 相同：返回已有 Revision 和当前状态，幂等成功。
3. 同一 `source_revision` 但 fingerprint 不同：返回 `409 SOURCE_REVISION_CONFLICT`；要求上游提供新修订，不能静默覆盖。
4. 新 `source_revision` 创建新的 Bundle Revision，初始为 `ACCEPTED`，随后由 Job 推进。
5. 新 Revision 达到 `READY`，或主信息与至少一个必需/可用 Document 已完成的 `PARTIAL` 后，在一个事务中更新 Bundle 的 `current_revision_id` 和 `availability_status`。
6. 新 Revision `FAILED` 时，Bundle 继续指向上一个 READY/PARTIAL Revision；没有历史可用版本时 Bundle 为 `FAILED`。

Revision 状态为：`ACCEPTED → PROCESSING → READY | PARTIAL | FAILED`；显式补传或重试可使 `PARTIAL/FAILED → PROCESSING`。来源 Manifest、附件声明、fingerprint 与来源修订不可修改；重试只创建新 Job 或补齐此前声明但未接收的成员。完整失败与恢复规则见[入库失败与恢复设计](11_step_1_ingestion_failure_and_recovery.md)。

## 与 Document Version 的关系

每次 Bundle Revision 都有一份不可变附件成员清单。后续以 `KBOT_KC_BUNDLE_REVISION_DOCUMENT` 记录该 Revision 包含的 `document_id` 与 `document_version_id`、角色、顺序和必需性：未变化附件可复用已有 Document Version，变更附件创建新 Document Version，移除附件只是不再属于新 Revision。这样来源快照、附件版本与最终 Evidence 都可准确回溯。

## 已确认

- `PARTIAL` 可切换为 Bundle 当前 Revision，条件是 Manifest 与至少一个必需或可用 Document 已完成。
- Bundle Revision 只保留规范化后的 Manifest JSON；原始上游 payload 如需保留，仅写入受控审计 URI，不进入 KC 事实表。

## 来源修订说明

首期 KM Asset 作为发布后不变更的来源，`source_revision` 主要用于重复投递幂等，不需要设计复杂更新流程。通用 Adapter 后续接入可继续使用同一 Revision 模型；同 revision/fingerprint 不一致时返回 `409 SOURCE_REVISION_CONFLICT`，避免静默覆盖来源快照。
