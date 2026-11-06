# AIOps 查询分页游标

日期：2026-11-06。API 与 Worker 使用相同的签名与有效期规则。

## 巡检结果查询失败原因

统一助手的巡检查询由 Worker 执行，调用运行查询服务读取最近巡检批次。
第一页最多返回 20 条；仓储多取一条判断后续页，存在后续页时即生成签名游标。
原 Worker 装配遗漏 Cursor Codec，即使调用方没有传入游标，也会在记录超过一页时抛出
“AIOps 查询 Cursor Codec 尚未配置”，导致本轮规划失败；这不是巡检执行本身失败。

## 配置与边界

- API 与 Worker 都通过 `bootstrap/cursors.py:create_cursor_codec` 创建编解码器，
  注入各自运行查询服务及变更查询服务，不改变分页上限或跳过游标签名。
- 密钥来自 `management.cursor_secret_env` 指定的环境变量，有效期来自
  `management.cursor_ttl_seconds`。生产环境缺少密钥，或配置的密钥不足 32 字节，
  会在打开数据库、HTTP 会话或后台任务之前拒绝启动；不记录密钥内容。
- 仅开发环境允许沿用已有派生规则，固定使用 API 的 `service_name` 作为派生输入，
  不使用 Worker 名称或副本 ID。因此此前 API 签发的游标仍可验证，跨进程结果一致。
- 游标继续绑定租户/业务空间、调用方、查询过滤器及过期时间，不绕过当前查询授权。
- 本修复不改数据库、Prompt 或公开接口，不修改既有失败轮次。部署代码并重启受影响
  的 AIOps Worker 后，用户重新提问以创建新轮次；未部署不能据单元测试声称线上恢复。

## 回归验证

`tests/unit/aiops_agent/test_aiops_query_cursor_bootstrap.py` 覆盖 API/Worker 真实生命周期
的无效配置早失败、两处查询服务的装配、跨进程签名一致性、已有开发游标继续有效、
调用方与过滤条件隔离，以及 21 条巡检批次的第一页和后续页查询。
