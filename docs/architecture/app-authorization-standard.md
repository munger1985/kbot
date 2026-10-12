# App 权限设计规范

## 适用范围

本规范适用于 KBot 所有业务 App。App 彼此独立，成员、角色、权限和 Domain 范围均以
`APP_ID` 为边界，任何 App 管理员都不能跨 App 读取或授权。只有平台唯一全局管理员可在
平台控制面维护 App 生命周期、初始管理员和平台用户的显式 App Grant；平台身份不会隐式
获得业务权限。

新增或修改 App 权限时，权限目录、三个内置角色、API、Oracle 规范 DDL、Bootstrap、
Schema Runner、测试和本文档必须在同一变更中同步，不允许由页面或初始化脚本单独维护
另一套角色含义。

## 统一角色

每个业务 App 必须且只能提供以下三个 ACTIVE 系统角色，编码和语义全局一致：

| 角色编码 | 中文名称 | 权限边界 | Domain 范围 |
|---|---|---|---|
| `app_user` | App 用户 | 日常使用权限 | `SELECTABLE`，必须明确选择 Domain |
| `app_manager` | App 业务管理员 | `app_user` 加业务配置与业务管理权限 | `ALL_APP_DOMAINS` |
| `app_admin` | App 管理员 | App 全部权限，包括成员与角色管理 | `ALL_APP_DOMAINS` |

系统角色不可创建同名自定义角色，不可编辑、停用或删除。角色权限必须由
`platform_core.authorization` 的权限元数据机械生成：`app_user` 获得 USER 层，
`app_manager` 获得 USER + MANAGER 层，`app_admin` 获得 USER + MANAGER + ADMIN 层。
业务代码和 SQL 不得各自解释层级。

`{app_id}:member_manage` 和 `{app_id}:role_manage` 固定属于 ADMIN 层，且禁止加入自定义
角色。这样只有 `app_admin` 能管理成员和角色，自定义角色不能形成新的委派管理员。

## 自定义角色

专业职责通过 App 内自定义角色表达，不再增加系统角色。自定义角色必须遵守：

- 主键和授权边界为 `(app_id, role_code)`，不能跨 App 引用或分配；
- 必须包含本 App 的 `use` 权限，只能选择权限元数据中 `custom_assignable=true` 的权限；
- 创建和修改者只能分配自己当前实际拥有的权限，禁止权限提升；
- 固定使用 `SELECTABLE`，成员绑定时必须显式选择至少一个 ACTIVE App Domain；
- 不能停用；下线只能物理删除。

角色列表必须返回 `member_count` 和 `row_version`。更新和删除必须提交
`If-Match: "rv-N"`；缺少条件头返回 `428`，版本不一致返回 `412`。物理删除按 Scope、
成员绑定、角色权限、角色定义的顺序在同一事务完成。已有成员无需先人工解绑；删除后这些
成员立即失去该角色提供的权限，接口返回 `affected_member_count`。系统角色和受保护初始
管理员仍不可删除。

## 权限元数据与审计

每项 App 权限必须声明：

- 稳定的 `permission_code` 和用户可读名称；
- 默认层级 `USER`、`MANAGER` 或 `ADMIN`；
- 是否允许加入自定义角色。

角色创建、更新和删除必须与 `KBOT_ACCESS_AUDIT_EVENT` 在同一事务写入，记录 Actor、
App、动作、角色编码、安全摘要、受影响成员数和时间。审计记录不引用角色外键，确保角色
物理删除后仍可追溯。摘要不得包含密码、Token、完整 DSN 或其他凭据。

## 既有数据升级

升级前运行只读审计：

```bash
PYTHONPATH=packages/platform_core/src \
  conda run -n cube python scripts/db/audit_app_authorization.py
```

报告只输出 PDB/Schema、角色、权限、成员与 Scope 聚合数和异常项，不输出用户名或凭据。
完成备份并停止会写授权表的 Main API 后，由 Schema Owner 使用 SQL Developer F5 执行
`database/oracle/operations/upgrade_app_authorization_roles.sql`。

迁移规则固定如下：

1. `user` 绑定合并为 `app_user`；原 `SELECTED_DOMAINS` 原样复制，原
   `ALL_APP_DOMAINS` 展开为迁移时全部 ACTIVE App Domain，避免权限范围丢失。
2. `app_admin` 保留成员绑定并重建为 App 全权限。
3. 新建 `app_manager` 并写入规范权限，不自动分配任何成员。
4. 旧专业系统角色原位转为自定义角色，角色编码、权限、成员和 Domain 范围均保留。
5. 若存在孤儿绑定、统一角色编码冲突，或旧专业角色持有 `member_manage` / `role_manage`，
   迁移拒绝猜测并整体回滚。
6. 脚本可重复执行，不修改账号、密码、API Key 或业务数据。

迁移后再次运行只读审计，所有 `findings.item_count` 应为 `0`，三个系统角色的
`missing_permissions` 和 `excess_permissions` 均应为空。

## 新 App 验收清单

- 注册完整权限元数据，且每项权限只有一个默认层级；
- 只创建 `app_user`、`app_manager`、`app_admin` 三个系统角色；
- `app_admin` 等于 App 全权限，成员与角色管理权限不可自定义；
- 自定义角色 API 实施 App 隔离、权限防提升、Scope 校验、ETag 和物理删除；
- 角色变更写入持久化审计；
- DDL、Bootstrap、Schema Runner、OpenAPI 和授权验收测试全部通过。
