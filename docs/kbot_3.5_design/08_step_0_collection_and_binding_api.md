# 步骤 0 详细设计：Collection 与 Binding API

本文定义 V2 Collection 的管理面契约。所有路径的 `domain_id` 都是受认证校验的 Scope Selector；`app_id` 从 `base.toml` 注入，永不由请求体提供。

## Collection 管理 API

Collection 由具有 Domain 管理权限的 APEX 用户创建和维护。建议使用不可变的 `collection_key` 作为稳定路由键，显示名称可随时修改。

| 操作 | V2 API | 规则 |
| --- | --- | --- |
| 创建 | `POST /api/v2/knowledge/domains/{domain_id}/collections` | 创建 ACTIVE Collection；`collection_key` 在 App/Domain 内唯一 |
| 列表 | `GET /api/v2/knowledge/domains/{domain_id}/collections` | 返回当前用户可管理/可见的 Collection 与绑定计数 |
| 详情 | `GET /api/v2/knowledge/domains/{domain_id}/collections/{collection_key}` | 包含状态、默认安全等级和安全的统计摘要 |
| 编辑 | `PATCH /api/v2/knowledge/domains/{domain_id}/collections/{collection_key}` | 仅允许改显示字段、默认安全等级、metadata、状态 |
| 停用/启用 | `PATCH` 设置 `status=DISABLED/ACTIVE` | DISABLED 禁止新入库与检索，保留数据和 Binding |
| 删除 | `DELETE /api/v2/knowledge/domains/{domain_id}/collections/{collection_key}` | 仅未被绑定时可请求；返回 `202` 与 `purge_job_id` |
| 清除状态 | `GET /api/v2/knowledge/domains/{domain_id}/purge-jobs/{purge_job_id}` | 返回 `DELETING/FAILED/SUCCEEDED` 与安全错误摘要 |

创建请求：

```json
{
  "collection_key": "delivery_assets",
  "display_name": "交付资产库",
  "description": "面向交付团队的资料",
  "default_security_level": 1,
  "metadata": {}
}
```

`collection_key` 使用 `^[a-z][a-z0-9_-]{1,63}$`，创建后永久不可修改。更名需求只改 `display_name`；若必须改变 key，应新建 Collection 并显式迁移 Bundle/Binding，避免 Portal 配置、链接和审计记录失效。

任何 API 均不接受 `app_id`、`domain_id`（除路径 Scope）、内部 ID、状态机内部字段或直接指定来源类型。Domain 不匹配或无权限时返回不泄露资源存在性的 `404`；非法状态转换返回 `409`。

## Agent–Collection Binding API

Agent 配置属于 Agent 领域，Binding 事实属于 KC。APEX 的 Agent 配置页面应调用主 API 的 V2 Agent 配置服务；该服务在保存/删除 Agent 时以服务身份调用 KC 内部 Binding API，而不是让浏览器直接写 Binding 表。

| 操作 | 内部 API | 规则 |
| --- | --- | --- |
| 绑定 | `PUT /internal/v2/knowledge/domains/{domain_id}/agents/{agent_id}/collections/{collection_key}/binding` | 幂等创建或启用 Binding |
| 解绑 | `DELETE /internal/v2/knowledge/domains/{domain_id}/agents/{agent_id}/collections/{collection_key}/binding` | 幂等撤销当前 Binding |
| 查询 | `GET /internal/v2/knowledge/domains/{domain_id}/agents/{agent_id}/collection-bindings` | 供 Agent 保存、删除、Skill 路由与审计使用 |

Binding 操作须校验：Agent 属于该 Domain、Collection 属于该 Domain、Collection 为 ACTIVE、调用方具备 Agent 配置服务身份。当前 Binding 的唯一键为 `(consumer_type, consumer_id, collection_id)`；App/Domain Scope 通过 Collection 自动关联。所有 ACTIVE Binding 平权，不存在默认 Collection、权重或检索参数。Collection 停用不会撤销已有 Binding；`KnowledgeRetrievalSkillV2` 在解析 Agent 检索范围时自动跳过 `DISABLED` Collection，重新启用后 Binding 自动恢复生效。

删除 Agent 时，Agent 配置服务必须先解绑全部 Collection，再删除 Agent。若跨服务调用中断，Agent 删除工作流必须重试解绑；KC 另提供受限的 Binding 对账任务，报告或清理已经不存在的 Agent 引用，防止孤儿 Binding 永久阻止 Collection 删除。重试超过上限时，具有受限运维权限的管理员可通过强制清理接口移除已验证的孤儿 Binding；该操作必须记录原 Binding、Agent 不存在的校验证据、执行者和原因。

## 删除与并发规则

`DELETE Collection` 在事务中锁定 Collection 并再次查询 ACTIVE Binding。并发 Binding 创建必须在同一锁或条件更新下检查 `status=ACTIVE`，因此不能在删除检查后重新绑定。成功进入 `DELETING` 后，删除请求幂等返回同一 `purge_job_id`；清除任务可安全重试。

APEX 直连数据库仅允许读取 `KBOT_KC_V_*` 视图，不能创建/修改 Collection 或 Binding。管理写操作必须经过 V2 API，以保证 Domain 校验、app_id 注入、审计和清除状态机完整执行。

## 已冻结规则

- `collection_key` 永久不可变。
- DISABLED Collection 保留 ACTIVE Binding，但 V2 检索自动跳过；重新启用后自动恢复参与检索。
- Agent 删除的解绑重试超过上限后，管理员可用受限运维接口强制清理经验证的孤儿 Binding，并保留完整审计。
