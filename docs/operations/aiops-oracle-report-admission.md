# Oracle 原生报告的功能准入

AWR、AWR Diff 和 ASH 仍仅支持 Oracle。功能入口的数据库类型筛选不变，
不得把其他数据库的统计报告当作 Oracle 原生报告。

## 准入规则

三个入口均由功能目录的 `required_entitlements` 声明
`ORACLE_DIAGNOSTICS_PACK`。菜单可用性、UI 选中目标后的提交校验和后台
固定规划共同遵守此声明。自然语言入口冻结同一份要求，确定目标后再次校验。

许可唯一依据是该目标持久化的 `capabilities_json.entitlements` 列表。
数据库连接成功、DBA 权限、只读查询能力或包函数可执行均不代表商业授权。
系统不自动添加许可，不通过扩大权限或移除 Executor 门禁恢复报告功能。
管理员必须核实实际 Oracle 授权后，通过目标管理流程维护声明。

多个 Oracle 目标并存时，菜单按可用目标汇总；某个已授权目标可用不代表其他
目标也已授权。选中具体目标后仍独立校验；可信资源上下文限定的目标集合不变。

## 错误与恢复

- `AIOPS_STARTER_ENTITLEMENT_REQUIRED`：当前目标缺少入口声明的商业许可。
  菜单显示不可用及原因，UI 提交返回明确的 422；后台规划请求直接失败，
  不执行诊断工具、不重复投递。管理员先核实实际授权，再维护目标声明。
- `AIOPS_STARTER_TOOLS_UNAVAILABLE`：当前能力目录缺少固定入口所需工具
  或 Playbook。检查数据库版本、只读连接、许可声明和部署的诊断目录。
  不把这一配置缺口包装成模型引用未注册工具或通用内部错误。

后台准入错误会将关联 Turn 和主 Run 收敛为失败终态，保留稳定错误码及中文原因。
日志阶段为 `CHECK_STARTER_ADMISSION`，安全细节只包含缺失许可、工具或
Playbook 标识；不展开可能包含用户输入或凭据的异常链。其他未预期规划异常
仍保留独立的内部错误处理，功能编译和目录选择有明确阶段名称。

## 验证范围

回归测试位于：

- `tests/unit/aiops_agent/test_aiops_starter_admission.py`
- `tests/unit/aiops_agent/test_aiops_conversation_starters.py`
- `tests/unit/aiops_agent/test_aiops_turn_retry.py`

覆盖三种报告的无许可、合法许可、许可字段格式错误、能力目录缺失、
自然语言和 UI 提交、Oracle 类型筛选、多目标汇总、固定 Tool DAG 参数校验、
Outbox 单次失败及 Turn/Run 终态。正常路径测试加载真实诊断目录，但不连接
Oracle、不调用原生报告包；线上报告产出仍需部署后在实际授权目标上验收。

本修复不涉及持久化 Schema、初始化、数据库迁移或 Prompt 版本变更。
