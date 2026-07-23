# Platform migrations

`001_platform_domain.sql` 创建 Main API 拥有的 Domain 注册表和 APEX
只读视图。4.0 运行时只读取 `KBOT_PLATFORM_DOMAIN`，不回退读取旧
`KBOT_MD_DOMAIN`。

正式切换前通过独立、可核对的数据迁移把仍有效的旧 Domain 映射到新表。迁移必须
保留 `APP_ID`、`DOMAIN_ID` 和启停状态，完成数量、唯一键和 Collection 引用校验后，
才能启用 Portal 流量。
