# 仓库结构

KBot 使用服务型单仓库。各服务共享一个 Oracle Schema，但源码、入口、依赖和
私有资源按可独立构建边界组织。

```text
services/       Main API、Knowledge Retrieval App、Agent Runtime、KC、AIOps、Data Query、模型服务
packages/       platform_core 与 platform_clients 共享包
database/       同一 Schema 下按所有者拆分的 Oracle DDL
configuration/  唯一部署配置及说明
resources/      进程拓扑等部署级不可变资源
integrations/   APEX 等外部系统交付物
tools/          不进入生产构建的开发工具
tests/          单元、集成、契约、验收、Smoke 与质量评估
scripts/        数据库、部署、安全和发布操作入口
var/            本地日志、上传数据和生成物，Git 整体忽略
```

服务进程入口统一位于
`services/<service>/src/<package>/entrypoints/`。开发环境先执行：

```bash
bash scripts/deployment/install_workspace.sh
```

开发环境把各内部包以 editable package 安装，生产环境构建并安装 Wheel。启动脚本
不注入源码路径；服务间调用仍只能经过稳定契约和客户端，不允许直接导入其他服务的
Entity、Repository 或应用服务。

`database/oracle/` 保持集中，是因为 4.0 使用同一 Schema 并需要统一初始化；
目录内部仍按表所有者拆分，包括 KM Asset 的 Slack Inbox/Outbox。将来拆库时可直接
把对应服务的 DDL 目录随服务迁走。

文档只保留当前架构、产品说明、部署指南和冻结契约，入口见
[`docs/README.md`](../README.md)。模型下载等可执行工具属于部署脚本，位于
`scripts/deployment/models/`，不得放入文档目录。
