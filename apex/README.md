# KBot 4.1 APEX 应用安装与使用说明

本目录用于存放 KBot 4.1 的 Oracle APEX 应用导出文件。本说明仅覆盖以下应用：App 1000、1002、1003、1004、1005。

## 一、安装前准备

安装前请确认：

1. Oracle APEX 与 ORDS 已正确安装并可正常访问；
2. 已准备好用于安装 KBot 应用的 APEX Workspace 和解析 Schema；
3. KBot Python 服务已部署，并能通过计划配置的地址正常访问；
4. 使用具有应用导入权限的 APEX 开发者账号执行安装；
5. 保持应用 ID 不变，避免平台管理应用与各 App Agent 之间的关联失效。

## 二、安装说明

请严格按照以下顺序安装。

### 1. 安装 App 1000

首先导入并安装 `f1000.sql`。

App 1000 是 **KBot Main API 连接配置与应急恢复**应用。导入时必须同时安装该应用导出文件中包含的 **Supporting Objects（支持对象）**，以创建 KBot APEX 运行所需的数据库对象和基础配置。

在 APEX 导入向导中，请确认：

- Application ID 为 `1000`；
- 安装目标 Workspace 和解析 Schema 正确；
- Supporting Objects 选择安装，不要跳过。

### 2. 安装 App 1002

App 1000 及其支持对象安装完成后，导入并安装 `f1002.sql`。

App 1002 是 **KBOT智能体管理平台**，负责管理 App Agent、平台账号与权限、业务域、模型配置以及 KBot Main API 连接等平台级配置。App 1002 不包含 Supporting Objects，只需安装应用本身。

### 3. 安装 App Agent 应用

最后根据实际需要安装 App Agent 应用：

- App 1003：知识中心；
- App 1004：AIOps；
- App 1005：知识资产。

App 1003、1004、1005 均为可选应用，可以只安装实际需要的应用。这些 App Agent 应用不包含 Supporting Objects，只需安装应用本身。当前目录已提供 App 1004 的导出文件 `f1004.sql`；安装 App 1003 或 App 1005 前，请先将对应的 APEX 应用导出文件放入本目录，再通过 APEX 导入向导安装。

推荐的完整安装顺序为：

```text
App 1000（含支持对象）
  → App 1002
    → App 1003（可选）
    → App 1004（可选）
    → App 1005（可选）
```

> 不建议先安装 App Agent，再补装 App 1000 或 App 1002。App 1000 提供基础连接配置和应急恢复能力，App 1002 提供 App Agent 与统一权限等平台管理能力。

## 三、使用说明

1. 使用 APEX 应用前，请先确认 KBot Python 服务已经正常启动，并且相关 API 可以访问。
2. 首次使用时，请使用 APEX 账户登录 App 1000，配置 KBot Python Main API 地址并验证连接。
3. 完成 Main API 配置后，使用 KBot 平台管理员账号登录 App 1002。App 1002 主要用于管理：
   - App Agent；
   - 平台用户及其权限；
   - 业务域；
   - 模型配置；
   - KBot Main API 连接。
4. 根据业务需要安装并使用 App 1003、1004、1005。未安装的可选应用不会影响 App 1000 和 App 1002 的基础管理功能。
5. App Agent 能否显示和访问，还取决于应用启用状态、用户权限以及业务域授权。安装应用后，应在 App 1002 中完成相应配置。

## 四、应用介绍

| App ID | 名称 | 概述 | 是否必须安装 |
| ---: | --- | --- | --- |
| 1000 | KBot Main API 连接配置与应急恢复 | 配置 KBot Python Main API 地址，提供首次连接配置及应急恢复入口；安装时必须同时安装 Supporting Objects。 | 是，最先安装 |
| 1002 | KBOT智能体管理平台 | KBot 平台管理入口，用于管理 App Agent、平台用户与权限、业务域、模型配置和 KBot Main API 连接。 | 是，在 1000 之后安装 |
| 1003 | 知识中心 | 面向知识采集、处理、管理和智能问答等知识中心业务的 App Agent。 | 可选 |
| 1004 | AIOps | 面向智能运维的 App Agent，提供运维工作台、业务系统、监控、诊断、巡检、运维知识库等能力。 | 可选 |
| 1005 | 知识资产 | 面向知识资产管理与使用的 App Agent。 | 可选 |

