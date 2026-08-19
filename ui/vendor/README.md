# 浏览器端第三方依赖

KM 页面没有前端构建步骤，因此把经过版本锁定的浏览器构建产物随静态页面发布：

- `marked.umd.js`：Marked 18.0.9，MIT License；
- `purify.min.js`：DOMPurify 3.4.13，Apache-2.0 或 MPL-2.0 License。

文件复用自同一工作区 Ammolite Portal 的锁定依赖，原始构建产物内保留了许可证声明。
升级时必须同步修改 `ui/km/chat.html` 的静态资源版本参数，并重新执行 KM UI 合同测试。
