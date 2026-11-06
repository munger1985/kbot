# Oracle 原生报告的功能准入

AWR、AWR Diff 和 ASH 仍仅支持 Oracle。功能入口的数据库类型筛选不变，
不得把其他数据库的统计报告当作 Oracle 原生报告。

## 准入规则

Agent 授权业务系统包含可用 Oracle 目标时，三个快捷入口开放，不使用商业许可
声明决定按钮是否可点。选择实例时仍只能选择 Agent 授权范围内的 Oracle，
非 Oracle 目标不能提交这些功能。已有的目标启用、只读连接及连通性检查不变。

UI 提交和自然语言入口都冻结目录的 `required_entitlements` 要求；确定目标后的
后台固定规划校验 `ORACLE_DIAGNOSTICS_PACK`，不把执行准入错误变成入口禁用。

许可唯一依据是该目标持久化的 `capabilities_json.entitlements` 列表。
数据库连接成功、DBA 权限、只读查询能力或包函数可执行均不代表商业授权。
系统不自动添加许可，不通过扩大权限或移除 Executor 门禁恢复报告功能。
管理员必须核实实际 Oracle 授权后，通过目标管理流程维护声明。

多个 Oracle 目标并存时，菜单按可用目标汇总；入口开放不代表已核实所有目标的
商业许可或包执行权限。选中具体目标后独立执行准入；可信资源上下文限定的
目标集合不变，也不因其他业务系统存在 Oracle 而扩大当前 Agent 的授权范围。

## 错误与恢复

- `AIOPS_STARTER_ENTITLEMENT_REQUIRED`：当前目标缺少执行所需的商业许可声明。
  快捷入口不置灰；后台规划返回明确的业务失败，
  不执行诊断工具、不重复投递。管理员先核实实际授权，再维护目标声明。
- `AIOPS_STARTER_TOOLS_UNAVAILABLE`：当前能力目录缺少固定入口所需工具
  或 Playbook。检查数据库版本、只读连接、许可声明和部署的诊断目录。
  不把这一配置缺口包装成模型引用未注册工具或通用内部错误。

后台准入错误会将关联 Turn 和主 Run 收敛为失败终态，保留稳定错误码及中文原因。
日志阶段为 `CHECK_STARTER_ADMISSION`，安全细节只包含缺失许可、工具或
Playbook 标识；不展开可能包含用户输入或凭据的异常链。其他未预期规划异常
仍保留独立的内部错误处理，功能编译和目录选择有明确阶段名称。

## 报告时间绑定

AWR 快照查询在 SQL 中使用 `SYS_EXTRACT_UTC` 明确返回真实 UTC 时间，不依赖
驱动为无时区值推断偏移。绑定器按实际时刻比较快照边界，不忽略 UTC 偏移，也
不把不同秒的时间按分钟截断。补全报告动作时，同一窗口的 `Z`、毫秒和 `+08:00`
写法按实际时刻去重，避免重复 AWR 动作干扰 AWR Diff 的窗口配对。

ASH 的浏览器输入支持 UTC `Z`、毫秒时间及带偏移的 ISO 8601 时间，在参数校验时统一为
UTC、六位小数秒和显式偏移，保留亚秒边界。SQL 使用匹配的
`YYYY-MM-DD"T"HH24:MI:SS.FFTZH:TZM` 格式；不再把浏览器时间直接交给只接受
秒级加空格偏移的旧模板。规划与 Executor 重复校验必须得到同一参数和哈希。

## 验证范围

回归测试位于：

- `tests/unit/aiops_agent/test_aiops_starter_admission.py`
- `tests/unit/aiops_agent/test_aiops_conversation_starters.py`
- `tests/unit/aiops_agent/test_aiops_discovery_binding.py`
- `tests/unit/aiops_agent/test_aiops_turn_retry.py`

覆盖三种报告的入口开放、执行许可校验、许可字段格式错误、能力目录缺失、
自然语言和 UI 提交、Oracle 类型筛选、多目标汇总、Outbox 单次失败及 Turn/Run
终态。正常路径加载真实目录，覆盖快照绑定、Tool DAG 编译、冻结工具版本、
签名 Grant、Executor 参数校验、HTML 结果归一化和产物下载。数据库边界使用模拟
驱动，不连接 Oracle、不调用原生报告包；线上产出仍需部署后在实际目标上验收。

本修复不涉及持久化 Schema、初始化、数据库迁移或 Prompt 版本变更。
