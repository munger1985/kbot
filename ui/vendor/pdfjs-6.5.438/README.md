# PDF.js 静态资源

直接检索阅读区使用上游 npm 包 `pdfjs-dist@6.5.438`（Apache-2.0），校验值为：

```text
sha512-vg8Pw1x+AFKy4AMDvhuWl5cOYV8d698UoEANoFFhc79hqXDnZQqnOXoxw3HG907WHOg1JYOvLhP756boddFxpA==
```

使用 `build/pdf.min.mjs` 和 `build/pdf.worker.min.mjs`，仅扩展名改为 `.js`，两端静态服务器按 JavaScript 提供。保留中文 CMap、标准字体、ICC 和图像解码资源及各自许可证。不加载 PDF 脚本功能，不包含 QuickJS 文件。升级时两端须同时替换静态资源和阅读器的版本路径。
