# KBot 4.0、KM 4.0 与 KBot 4.1 分线规则

> 状态：`km4.0` 与 `kbot4.1` 已从共同基线提交
> `7dad2a49902d6be97582b0069d44fba07b95e3aa` 建立；KBot 4.1 已完成 KM 物理删除。
> `v4.0` 与 `km-v4.0` 标签尚未创建，数据库升级、部署和在线验收不属于本次提交。

## 1. 强制决策

共同基线提交同时作为两个终版基线：

- 主产品分支 `kbot4.1` 已建立，后续可在共同基线补充 KBot 4.0 标签 `v4.0`；
- KM 独立维护分支 `km4.0` 已建立，后续可在共同基线补充 `km-v4.0` 标签。

补充标签时，`v4.0` 与 `km-v4.0` 必须指向上述同一提交，不得指向 4.1 删除提交。分线后：

1. Knowledge Answering 与 AIOps 本轮改造只在 `kbot4.1` 实施。
2. `kbot4.1` 在同一改造中物理删除 KM App、KM 集成、KM Schema 和 KM 测试，不保留禁用开关、兼容路由、
   dead code 或可选部署模块。
3. KM 后续缺陷从 `km4.0` 创建修复分支，只合回 `km4.0`，按需发布 `km-v4.0.x`。
4. KM 修复不得 merge、rebase 或 cherry-pick 到 `kbot4.1` 及后续分支；4.1 功能也不得回灌 `km4.0`。
5. 两条产品线使用独立构建物、共享服务版本和 Oracle Schema，不得跨线混合部署或调用。

## 2. 终版标签补充门禁

补充终版标签前必须：

1. 将批准纳入终版的代码、DDL、OpenAPI、页面和文档提交到 `kbot4.0`；工作树内容不能直接成为标签。
2. 在规范 Oracle Schema 上完成 KBot 4.0 全量测试和受控在线验收，包括 KM、Slack、Qn/Cn、引用预览、
   Runtime、KC、Main API、静态 UI、初始化和升级检查。
3. 记录批准提交 SHA、Schema manifest 版本、构建物摘要、测试结果和已知限制。
4. 在同一提交创建 annotated tag `v4.0` 与 `km-v4.0`；标签不得移动或重打。
5. 核对现有 `kbot4.1` 与 `km4.0` 的共同祖先就是该批准提交，再推送标签并保护两个分支。

实际 tag 和 push 属于发布操作，必须在用户明确授权后执行。

## 3. KM 独立维护线

```text
km-v4.0
  -> km4.0
       -> fix/km-<issue>
       -> KM 4.0 全量回归
       -> merge to km4.0
       -> optional km-v4.0.x
```

`km4.0` 保留 KM 正常运行所需的完整 4.0 技术栈，而不只是 `km_asset_app` 目录，包括固定版本的 Main API、
Agent Runtime、Knowledge Core、Data Query、Model Serving、Oracle Schema、UI 和 Slack。需要修改共享服务时，
修改只存在于 `km4.0`，不承担向 KBot 4.1 同步的义务。

若同一安全问题同时影响两条线，必须分别基于 `km4.0` 与当期 KBot 分支实现、审查和验收两个独立修复，
不能复用提交或跨线合并。

## 4. KBot 4.1 一次性删除范围

删除不是逐步下线。`kbot4.1` 的 Knowledge Answering 改造提交必须同时清除以下内容及所有引用：

| 删除面 | 典型路径或内容 |
| --- | --- |
| KM 服务 | `services/km_asset_app/**`、workspace package、API/Worker/Slack Worker |
| KM UI | `ui/km/**`、共享 Shell 中的 KM 导航/认证、Dev Console KM 静态路由 |
| Main API | `api/km_asset_app.py`、`api/slack.py`、KM 登录/API Client/路由注册和客户端装配 |
| Agent Runtime | `specialists/km_asset/**`、KM Route Planner、AppScoped KM 注册、KM Prompt/合同 |
| Knowledge Core | `KM_ASSET` Intake、`km-assets` 内部路由、KM Preview/Manifest 特例 |
| Data Query | KM managed dataset、Asset 查询合同和 KM 平台访问特例 |
| Platform packages | `platform_clients/km_asset.py`、`km_portal.py`、KM/Slack 专属合同与配置 |
| Oracle | `database/oracle/km_asset_app/**`、KM bootstrap、权限/角色种子、manifest 注册 |
| 运行与发布 | topology、start/stop、安装清单、workspace fingerprint、release verifier、配置示例 |
| 测试与文档 | KM 单元/集成/验收/合同测试、KM OpenAPI、KM/Slack 产品和运维文档 |

Slack 只有在确认完全属于 KM 时随 KM 删除；若审计发现非 KM 消费者，先将通用合同与 KM 适配器分开，
但不得为 KM 在 4.1 保留入口。任何保留项都必须由非 KM 调用方和测试证明，不能因名称“可能通用”而留下。

## 5. 4.1 删除完成定义

`kbot4.1` 必须以正向功能测试和负向不存在性检查共同验收：

- `rg` 与依赖图中不存在 `km_asset`、`KM_ASSET`、`km-assets`、`ui/km` 或 KM Slack 路由的可执行引用；
- workspace、安装目录、拓扑和发布制品不包含 KM 包或进程；
- Main API/OpenAPI 不包含 KM、KM 登录或 KM Slack Callback；
- Runtime 不注册 KM Planner、Skill、Composer 或 `owner_app_id=km_asset` 分支；
- KC/Data Query 不保留 KM 专用 Intake、查询或资源类型；
- canonical Oracle DDL、bootstrap、manifest、角色与权限种子不创建 KM 对象；
- UI 构建物、配置示例、监控和运维文档不再宣称 KM 属于 KBot 4.1；
- Knowledge Retrieval、AIOps、Runtime、KC、Data Query、Main API 和发布检查全部通过。

删除后的 4.1 不再运行 KM 功能回归；KM 功能回归只属于 `km4.0`。4.1 门禁验证的是 KM 已彻底不存在，
而不是“虽然保留但行为不变”。

## 6. 与 Knowledge Answering 改造的关系

分线消除了 KM 对 4.1 共享 Runtime/KC 重构的兼容约束：

- 4.1 的 Knowledge Answering 只服务 Knowledge Retrieval、AIOps 和未来 App；
- 4.1 不保留 `AppScopedSkill` 的 KM 分支、KM Execution Spec、Qn/Cn 或 Slack 兼容路径；
- KM 继续使用 `km4.0` 固定技术栈，并独立接受 `km-v4.0.x` 修复；
- Ammolite 不受该 Git 分线影响，也不引入 KM App 概念。

## 7. 分支保护

- 保护 `km4.0` 和 `kbot4.1`，禁止直接推送；所有修改通过对应目标分支的 PR。
- `km4.0` PR 必须运行 KM 4.0 全量回归，并拒绝 4.1 合同、Schema 和产品功能。
- `kbot4.1` PR 必须运行 Knowledge Answering/AIOps 全量回归和 KM 不存在性检查。
- CI 拒绝 `km4.0` 与 `kbot4.1` 之间的 merge commit、cherry-pick 标记和 KM 路径重新引入。
- 发布记录明确标识 `KM 4.0.x maintenance` 或 `KBot 4.1+ active product`，不得共用同一部署制品。
