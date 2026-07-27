# 共享 Python 包

这里只存放两个或以上服务必须复用、且具有稳定边界的 Python 包。

- `platform_core`：配置、数据库、日志、鉴权、公共契约和基础持久化能力。
- `platform_clients`：服务间 HTTP 客户端，不包含被调用服务的业务逻辑。

服务私有代码不得为了复用方便下沉到这里。开发环境使用可编辑安装：

```bash
pip install --no-deps -e packages/platform_core -e packages/platform_clients
```
