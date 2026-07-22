# 步骤 5 详细设计：Discovery Profile 基础

本阶段建立第一阶段检索的数据边界：`KBOT_KC_DISCOVERY_OBJECT` 同时支持 `BUNDLE` 与 `DOCUMENT` 两种画像，归属于不可变 Bundle Revision。它只回答“哪个业务对象/文件值得继续查”，不作为事实引用；最终回答仍必须来自 Evidence。

## 已落地内容

- `KcDiscoveryObjectEntity` 与增量迁移 `006_kc_discovery_object.sql`。
- `DiscoveryRepository`：按 Revision/Profile Key 幂等写入、激活和历史画像撤销。
- `build_bundle_profile()`：不调用 LLM 的确定性画像拼接，包含标题、来源、Facet、成员角色/MIME/状态、Evidence 覆盖和缺失成员；同一输入产生相同 `profile_hash`。
- Profile 不携带 `app_id/domain_id`，通过 Collection 继承 APEX/Domain Scope。
- `PROFILE` 完成后自动投递目标为 `DISCOVERY` 的 `INDEX` Job；它使用同一 Collection 模型生成 Bundle/Document 画像向量。

## 状态边界

`PARSE → INDEX` 完成后，Revision 才能创建 `PROFILE` Job。Profile 生成 `STAGED` 画像；目标为 Discovery 的 INDEX 成功后才激活画像并切换 Bundle `current_revision_id`。历史 Revision 的画像即使未物理清理，也不能被当前版本查询召回。

本阶段尚未接入 Oracle Text/Vector 查询和 LLM 候选选择器；下一步将实现 Bundle 级候选聚合与两阶段查询 API。
