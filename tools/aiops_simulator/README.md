# KBot AIOps数据库流量模拟程序

这是一个与KBot服务、`scripts/aiops-stack`和数据库容器完全解耦的独立Python测试
程序。它只在人工执行`start`后作为宿主机后台进程运行，不使用Docker、不安装systemd
服务、不设置开机自启，也不会启动、停止或重启任何KBot组件。

## 日常流量能力

- 使用`Asia/Shanghai`真实时间，不压缩一天；
- Oracle ERP、PostgreSQL MES、MySQL WMS各自使用独立的分时流量曲线；
- 每100个逻辑事务精确包含80个读事务和20个写事务，窗口内随机打散；
- 使用连接池、绑定参数和稳定SQL模板，便于AWR、Performance Schema和
  `pg_stat_statements`形成可解释的工作负载；
- 历史查询读取既有2025年千万行分区表；
- 写流量进入每库独立的实时活动表，不改变三张精确1000万行的基准表；
- 过期实时活动数据按小批量清理，默认保留30天；
- 连续失败时自动退避，不制造连接风暴。

故障模拟是同一工具内的独立进程，不会改变或依赖`daily`进程是否运行。

## 目录与运行边界

- 唯一配置：`var/aiops-simulator/simulator.ini`
- PID文件：`var/aiops-simulator/simulator.pid`
- 日志文件：`var/aiops-simulator/simulator.log`
- Python解释器：默认`python3`，可通过`AIOPS_SIMULATOR_PYTHON`指定
- 网络：仅由测试程序主动连接配置中的数据库地址，不监听或发布任何端口

实际配置包含数据库密码，`manage`要求权限为`0600`或`0400`。密码不会进入命令行、
日志或Git。

模拟器会写入自有活动表并为故障场景持有事务，因此三个密码必须属于业务Schema账号：

- Oracle：`AIOPS_TEST`，即ERP对象所有者；
- PostgreSQL：`aiops_test`，即`mes` Schema所有者；
- MySQL：`aiops_test`，即`aiops_demo`业务应用账号。

不得复用AIOps Target的`AIOPS_DIAG`或`aiops_diag`诊断密码。Oracle和PostgreSQL会核对
当前账号与Schema所有者，MySQL会核对业务库级完整读写及DDL权限；身份不符合时，
`prepare`、日常流量和故障模式都会拒绝运行。

运行环境需要Python 3.10或更高版本，并安装`requirements.txt`中的三个数据库驱动。
KBot4标准Python环境已经包含这些固定版本；如果使用独立虚拟环境，可执行：

```bash
python3 -m pip install -r tools/aiops_simulator/requirements.txt
```

## 部署时生成运行配置

正式环境不要求演示人员再次填写密码。三库部署/装填数据流程在生成业务账号密码后，必须
调用配置生成入口，直接读取权限受控的密码文件：

```bash
PYTHONPATH=tools/aiops_simulator python3 -m aiops_simulator render-config \
  --template tools/aiops_simulator/simulator.example.ini \
  --oracle-password-file /opt/aiops-oracle19c/secrets/aiops_test_pwd \
  --postgresql-password-file /opt/aiops-databases/secrets/postgres_app_password \
  --mysql-password-file /opt/aiops-databases/secrets/mysql_app_password \
  --output /opt/aiops-demo-data/aiops-simulator/simulator.ini
```

生成器不会把密码写入命令行或日志，要求三个源密码文件权限不宽于`0600`，并以原子替换
方式生成`0600`配置。数据库部署流程随后把该文件安全安装到kbotdev的
`var/aiops-simulator/simulator.ini`；该传递过程不得经过Git或普通日志。

`manage init-config`只为没有数据库部署流程的本地开发生成空白模板，不是正式部署步骤。

## 首次准备数据库对象

```bash
tools/aiops_simulator/manage prepare
tools/aiops_simulator/manage validate
```

`prepare`是唯一会执行DDL的命令，创建以下三张日常活动表及普通索引：

- Oracle：`AIOPS_SIM_ERP_ACTIVITY`
- PostgreSQL：`mes.aiops_sim_mes_activity`
- MySQL：`aiops_sim_wms_activity`

同时创建三库各自的`aiops_sim_fault_account`故障账户表（Oracle名称为大写），并写入
两条固定测试记录。锁等待、死锁和长事务只操作这些模拟程序自有记录，不会锁定或修改
业务表。

命令具有幂等检查，不删除或修改既有业务表。普通`start`只做结构预检，缺表或列不匹配
时拒绝启动，不会自动修复Schema。

## 手动启停

```bash
tools/aiops_simulator/manage start
tools/aiops_simulator/manage status
tools/aiops_simulator/manage logs
tools/aiops_simulator/manage stop
```

宿主机重启后程序保持停止，必须再次人工执行`start`。`stop`只向PID文件确认过身份的
模拟进程发送`SIGTERM`；程序停止调度、等待在途事务并关闭三个连接池。35秒内没有退出
时，管理脚本只报告错误，不会发送`SIGKILL`，也不会停止数据库或观测组件。

## 默认真实时间流量

配置示例分别体现ERP办公业务、MES全天生产和WMS出入库峰值。默认单库峰值不超过
4个逻辑事务/秒，周末按各系统不同倍率降载。所有时间段、周末倍率、并发和保留期都在
唯一INI中维护，但`daily`模式的8:2读写比例是程序合同，不允许通过配置改成其他比例。

启动日志每分钟报告各数据库累计读写数、实际比例、失败数、平均耗时和主要SQL模板。
数据库异常只会让对应Runner退避，其他数据库继续运行。

## 故障模拟

故障命令不使用单独的配置文件或配置段。启动时直接传入数据库和故障类型，程序只从既有
`simulator.ini`读取目标数据库连接信息：

```bash
tools/aiops_simulator/manage fault oracle start blocking_lock
tools/aiops_simulator/manage fault postgres start slow_query
tools/aiops_simulator/manage fault mysql start connection_surge

tools/aiops_simulator/manage fault oracle status
tools/aiops_simulator/manage fault oracle stop
```

三个数据库均支持以下故障类型：

- `slow_query`：在千万行基准表上执行不可索引的聚合扫描；
- `blocking_lock`：在故障账户表上建立一个阻塞者和一个等待者；
- `deadlock`：在故障账户表的两行之间周期性制造反向加锁；
- `long_transaction`：更新故障账户表后保持事务不提交；
- `connection_surge`：建立12个带运行标识的独立连接并保持；
- `temp_pressure`：在千万行基准表上执行需要临时空间的大排序。

每个数据库同时只允许一个故障进程。所有故障默认最多运行10分钟，到期自动回滚事务并关闭
本次创建的连接；`stop`执行同样的定向恢复。管理脚本只识别自身PID、命令行目标和状态
文件，不会终止其他数据库会话，也不会停止数据库或容器。连接数、持续时间、扫描规模和
临时空间压力均为代码内置的保守值，不提供命令行调大入口。
