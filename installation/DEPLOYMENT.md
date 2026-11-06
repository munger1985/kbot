# KBot 一份配置、一键安装

保留独立业务镜像，由一个零参数入口完成镜像准备、解析模型安装、首次数据库和账号初始化、
启动与基础验收。目标机不需要源码、Git、Conda、pip 或本地构建镜像。

## 用户安装

准备 Linux x86_64、Python 3.11+（仅标准库）、Docker Engine 和支持 `up --wait` 的
Docker Compose v2。Docker 必须是本机 Linux x86_64 服务端。提前由 DBA 准备专用
Oracle PDB/Schema、ASSM 默认表空间和额度，以及 `CREATE SESSION`、`CREATE TABLE`、
`CREATE VIEW`；选择 Knowledge Core 时还需要 `EXECUTE ON SYS.DBMS_ALERT`。
初始化前建立数据库恢复点。入口不会安装宿主机软件，也不会创建 PDB 或授予 DBA 权限。

取得并验证版本化部署包的外部 SHA-256，解压后执行：

```bash
./kbot-deploy
```

首次执行生成权限为 `0600` 的 `deployment.ini`，不会连接数据库或启动容器。
填写 Oracle 地址和密码、对外访问地址、数据/日志路径、App 选择。
只有明确确认空库和恢复点后，才将 `initialize_empty_schema` 改为 `true`。
私有仓库先通过 `docker login` 登录，仓库认证信息不写进 KBot 配置。

```bash
vi deployment.ini
./kbot-deploy
```

这就是目标机的完整安装入口。内部端口、依赖、Compose、运行时 TOML 和 Secret 自动生成。
Registry、版本、镜像 Digest 与模型规格由发布包确定，用户不维护第二份清单。

在线包自动按 Digest 拉取所选 App 所需的业务镜像、UI 和 Installer。离线包自动
`docker load` 并按 Image ID 校验，全程 `--pull never`，不尝试访问 Registry。
解析模型使用包内受 SHA-256 保护的版本化资产，复用原有 Docling 模型安装器；不在首次
业务请求时临时下载。模型归档与业务镜像分开交付，不合并业务服务容器。

## 持久化与重复部署

数据目录下的 `.kbot/` 保存原主密钥、私有 Secret、安装记录和生成的编排快照。
不要删除它，也不要把它提交 Git。升级或迁移必须继续使用原数据目录和主密钥。
每次生成新的编排快照，不删除正在运行的容器所绑定的旧配置文件。

首次安装为平台 ADMIN 和所选 App 管理员生成独立随机密码，保存至
`<data_dir>/.kbot/initial-accounts.json`（`0600`），不打印到日志。
首次登录必须修改密码，完成交接后可受控移走该文件。

再次执行同一入口会保留数据库、密码和主密钥，不重放 DDL、Prompt 或 App 种子。
即使配置仍为 `initialize_empty_schema = true`，也不会重新初始化。
安装记录与数据目录、数据库目标、App 范围、向量维度、模型资产、镜像锁或主密钥不一致时拒绝自动部署；
新版本必须有对应的受控升级和备份流程，本入口不会把换镜像当作数据库升级。

初始化执行前写入 `initializing` 记录，完成后才记录 `initialized`；
启动、HTTP 探针、后台进程稳定性和登录页验收后才记录 `ready`。
Oracle DDL 隐式提交：初始化中断或记录不明时停止自动重试，由 DBA 检查或恢复空白 Schema，
不能删除记录后直接重放，也不会自动清库、撤销 DDL 或重置已有用户。
初始化完成但服务启动失败可以直接重试，不再初始化。

接管已有部署只适用于人工确认的同版本、同 App 范围：先恢复原主密钥到
`<data_dir>/.kbot/master_key`，关闭空库初始化并启用 `adopt_existing_schema`。
安装器只读检查规范对象、平台和所选 App 基础账号/授权，不执行种子或修改密码。

## 可用性边界

业务 HTTP 进程使用各自已有的 `/health`、`/ready` 或 `/readyz`；数据库 readiness
不增加 Schema 探测。无 HTTP 的 Worker 验收范围是容器存活及 10 秒内未重启，
不能据此宣称业务任务已成功执行。所有所选 App 的登录页都必须可访问。
模型 Provider、知识库、Agent、AIOps Target 和外部系统接入继续在 App 内配置，
基础安装成功不等于模型推理或端到端诊断业务验收成功。

AIOps 变更开关默认关闭。观测栈仍是独立 Compose Project，使用同一配置中的
`[observability]` 和 `[observability.*]`，并复用规范观测部署资源。
需要源码构建的 Exporter Wrapper、Collector 或签名桥会在部署前明确拒绝，
直到发布对应的预构建镜像；用户安装机不执行 `build`。
核心离线包没有第三方观测镜像，因此必须保持观测栈关闭。OEM 在 App 内配置。

## 发布人员制作部署包

先完成新镜像构建发布，并更新对应正式版本的镜像锁和 Docling 模型规格。
镜像必须包含新的 `scripts/db/initialize_installation.py` 和 `installation/catalog.toml`。
现存 1.0 镜像及归档不会自动获得此能力，不能搭配新的安装工具冒充新版本。
制作在线包要求干净工作区、工具/拓扑/模型规格/运行代码与镜像源码一致；
允许在镜像构建后单独提交 Digest 锁，但不能同时夹带运行代码变化：

```bash
installation/dev/kbot-release export-deployment \
  --config installation/releases/<版本>.ini \
  --docling-models-dir /srv/release-assets/docling_models \
  --output installation/generated/online/kbot-<版本>
```

包内只有部署控制器、规范拓扑/目录、唯一配置模板、模型资产和校验清单，
不含业务源码、数据库密码、主密钥和镜像归档。模型随包交付，因此包大小包含模型本身。
离线包继续使用 `export-offline` 制作，新增相同的 `kbot-deploy` 入口和模板，
额外包含业务镜像归档。`kbot-offline` 的分步命令仅用于运维诊断，普通安装不需要执行。

对发布包做整体归档并独立分发外层 SHA-256；包内 `SHA256SUMS` 用于完整性检查，
不替代可信发布渠道或签名。实际用户配置和所有状态都在发布后生成，不进入制品。
