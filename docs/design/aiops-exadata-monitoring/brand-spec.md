# 资产与样式来源

- 现有产品：Ammolite / AIOps，生产 Web 资产已由正常会话读取，不做外部品牌推断。
- 标识：apps/portal-web/public/favicon.svg，原样 data:image/svg+xml 内嵌，不重绘。
- 样式：apps/portal-web/src/shared/ui/styles.css，Noto Sans SC / IBM Plex Mono、--accent #315b78、--canvas #f2f5f8、--ink #17212b；暗色同一文件定义。
- 真实界面参照：/tmp/exadata-hardware-render-ammolite-Dl9HK6/overview.png。
- 内容为数据工具，不需要实体 Exadata 产品照；不使用机柜剪影、装饰性插画或第三方品牌 logo。
- 三个样稿均为内部内容区，不改变生产顶栏、账号菜单和全局导航。按钮实施时必须使用共享语义样式，设计稿中样式只作用于离线样稿。
- 资产自检：favicon 已核对为真实 SVG；正常浏览器截图有七个确认对象和四张数值卡；主色由真实源码提取，非凭空选色。
