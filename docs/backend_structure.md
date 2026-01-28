## KBot 后端架构文档

### 项目概述

KBot (Knowledge Based Chatbot) 是一个基于 FastAPI 的知识库问答系统，集成了大语言模型(LLM)、向量嵌入、重排序和视觉语言模型(VLM)等功能。系统采用模块化微服务架构，支持 Oracle 数据库和 Elasticsearch 向量存储。

### 目录结构

```bash
kbot3/
├── api/                        # API 层
│   ├── controllers/            # 控制器，处理业务逻辑
│   │   ├── agent_controller.py    # Agent 智能体控制器
│   │   ├── auth_controller.py     # 认证授权控制器
│   │   ├── kb_controller.py       # 知识库控制器
│   │   ├── log_controller.py      # 日志控制器
│   │   └── model_controller.py    # 模型控制器
│   ├── routers/                # 路由定义
│   │   ├── agent_router.py        # Agent 路由
│   │   ├── auth_router.py         # 认证路由
│   │   ├── health.py              # 健康检查
│   │   ├── kb_router.py           # 知识库路由
│   │   ├── log_router.py          # 日志路由
│   │   └── model_router.py        # 模型路由
│   └── schemas/                # Pydantic 数据模型
│       ├── accessor_schema.py     # 访问器模式
│       ├── agent_schema.py        # Agent 数据模型
│       ├── auth_schema.py         # 认证数据模型
│       ├── base_response.py      # 响应基类
│       ├── eslog_schema.py        # ES 日志模型
│       ├── kb_schema.py           # 知识库数据模型
│       ├── model_schema.py        # 模型数据模型
│       └── parser_schema.py       # 解析器数据模型
│
├── core/                       # 核心模块
│   ├── auth/                   # 认证授权
│   │   ├── entities/              # 用户和令牌实体
│   │   ├── repositories/          # 认证数据仓库
│   │   ├── auth_service.py        # 认证服务 (JWT/API Key/密码)
│   │   ├── dependency.py          # FastAPI 依赖注入
│   │   └── shortcuts.py           # 快捷方法
│   ├── config/                 # 配置管理
│   │   └── settings.py            # Dynaconf 配置类
│   ├── database/               # 数据库连接
│   │   ├── meta_oracle.py        # Oracle 元数据 (SQLAlchemy)
│   │   ├── vec_oracle_pool.py    # Oracle 向量连接池
│   │   └── vec_elasticsearch.py  # Elasticsearch 向量客户端
│   ├── logger/                 # 日志管理
│   │   ├── manager.py             # 日志管理器 (Loguru)
│   │   └── __init__.py
│   ├── middleware/             # 中间件
│   │   └── log_middleware.py      # 请求日志中间件
│   ├── dictionary.py           # 枚举和常量定义
│   └── exceptions.py           # 自定义异常
│
├── dao/                       # 数据访问层
│   ├── entities/               # 数据库实体定义
│   │   ├── base.py                # 实体基类
│   │   ├── kbot_md_*.py           # 元数据表实体
│   │   └── kbot_biz_*.py          # 业务表实体
│   └── repositories/           # 数据仓库
│       ├── common.py              # 通用仓库
│       ├── kbot_md_*_repo.py     # 元数据仓库
│       ├── kbot_biz_*_repo.py     # 业务仓库
│       ├── kbot_biz_txt_embedding_factory.py  # 嵌入仓库工厂
│       ├── kbot_biz_chat_session_factory.py  # 会话仓库工厂
│       └── embedding_repo/        # 嵌入仓库实现
│           └── [es|oracle].py     # ES/Oracle 实现
│
├── services/                  # 业务服务层
│   ├── chat/                   # 聊天服务
│   │   ├── agent_chat.py         # Agent 智能体 (传统实现)
│   │   ├── mcp_chat.py           # MCP Agent (新实现)
│   │   ├── agent_dify.py         # Dify 集成
│   │   ├── agent_params.py       # Agent 参数
│   │   └── agent_rerank.py       # 重排序
│   ├── kb/                     # 知识库服务
│   │   ├── kb_file_operator.py   # 文件操作
│   │   ├── kb_file_preview.py    # 文件预览
│   │   ├── kb_chunk_operator.py  # 文本分块
│   │   ├── kb_procedure.py       # 知识库过程
│   │   └── file_transformer.py   # 文件转换
│   ├── search/                 # 搜索服务
│   │   ├── kb_search.py          # 知识库搜索
│   │   ├── kb_search_for_mcp.py  # MCP 搜索工具
│   │   └── fulltext_preprocessor.py # 全文预处理
│   ├── dataparse/              # 数据解析服务
│   │   ├── file_parser_manger.py # 解析器管理器
│   │   ├── file_parser_service.py # 解析器服务
│   │   ├── file_processor.py     # 文件处理器
│   │   ├── parser_common.py      # 解析器通用功能
│   │   ├── txt_to_md.py          # 文本转 Markdown
│   │   ├── summary_parser.py     # 摘要解析
│   │   └── disposed/             # 已废弃的解析器
│   └── sys/                    # 系统服务
│       └── eslog_service.py      # ES 日志服务
│
├── microservices/             # 微服务层
│   ├── common/                # 通用模块
│   │   ├── model_entity.py        # 模型实体
│   │   ├── model_pool.py          # 模型池
│   │   └── model_repo.py          # 模型仓库
│   ├── llm/                   # LLM 微服务
│   │   ├── llm_service.py         # LLM 服务
│   │   ├── model_pool.py         # LLM 模型池
│   │   ├── model_factory.py      # 模型工厂
│   │   ├── schema.py             # LLM Schema
│   │   └── model/                # LLM 实现
│   │       ├── base.py
│   │       ├── oci_client.py
│   │       └── openai_client.py
│   ├── embedding/              # 嵌入微服务
│   │   ├── embed_service.py      # 嵌入服务
│   │   ├── model_pool.py         # 嵌入模型池
│   │   ├── model_factory.py      # 模型工厂
│   │   ├── schema.py             # Schema
│   │   └── model/                # 嵌入实现
│   │       ├── base.py
│   │       ├── bge_local.py
│   │       ├── cohere_client.py
│   │       ├── openai_client.py
│   │       └── qwen3_local.py
│   ├── reranker/              # 重排序微服务
│   │   ├── reranker_service.py   # 重排序服务
│   │   ├── model_pool.py         # 模型池
│   │   ├── model_factory.py      # 模型工厂
│   │   ├── schema.py             # Schema
│   │   └── model/                # 重排序实现
│   │       ├── base.py
│   │       ├── bge_local.py
│   │       ├── cohere_client.py
│   │       ├── openai_client.py
│   │       └── qwen3_local.py
│   ├── vlm/                   # 视觉语言模型微服务
│   │   ├── vlm_service.py        # VLM 服务
│   │   ├── model_pool.py         # 模型池
│   │   ├── model_factory.py      # 模型工厂
│   │   ├── schema.py             # Schema
│   │   └── model/                # VLM 实现
│   │       ├── base.py
│   │       └── openai_client.py
│   └── docparser/              # 文档解析微服务
│       ├── docling_service.py    # Docling 解析服务
│       ├── parser_schema.py      # Schema
│       └── parsers/              # 解析器实现
│           └── docling_parser.py
│
├── mcp_tools/                 # MCP 工具
│   ├── base.py                  # 工具基类
│   ├── calculator_tool.py       # 计算器工具
│   ├── internet_search_tool.py  # 网络搜索工具
│   └── kb_search_tool.py        # 知识库搜索工具
│
├── utils/                     # 工具类
│   ├── common.py                # 通用工具
│   ├── encoder.py               # 编码器
│   ├── model_client.py          # 模型客户端
│   ├── oracle_vec_handler.py    # Oracle 向量处理器
│   ├── parser_client.py         # 解析器客户端
│   ├── sanitize.py              # 数据清洗
│   └── serializer.py            # 序列化
│
├── configuration/              # 配置文件
│   ├── base.toml                # 基础配置
│   ├── development.toml         # 开发配置
│   ├── production.toml          # 生产配置
│   ├── custom_dict.txt          # 自定义字典
│   └── stopwords.txt            # 停用词
│
├── apex/                      # APEX UI 脚本
│
├── docs/                      # 文档
│   ├── backend_structure.md     # 后端架构 (本文档)
│   ├── kbot_ddl_v1.0.sql       # 数据库 DDL
│   └── install/                 # 安装脚本
│
├── tests/                     # 测试
│
├── knowledge_base/             # 知识库存储
│
├── logs/                      # 日志目录
│
├── kbot_main.py              # 主程序入口
├── kbot_app_embedding.py     # Embedding 独立服务
├── kbot_app_llm.py          # LLM 独立服务
├── kbot_app_reranker.py     # Reranker 独立服务
├── kbot_app_vlm.py          # VLM 独立服务
├── kbot_app_parser.py       # Parser 独立服务
├── requirements.txt         # Python 依赖
├── start_kbot.sh            # 启动脚本
└── stop_kbot.sh             # 停止脚本
```

---

## 核心组件说明

### 1. 配置管理

使用 **Dynaconf** 管理配置，支持多环境配置文件。

```python
# core/config/settings.py
class AppConfig(BaseModel):
    title: str
    version: str
    file_storage: str
    log: LogConfig

class OracleConfig(BaseModel):
    username: str
    password: str
    host: str
    port: int
    service_name: str

class LLMConfig(BaseModel):
    service_host: str
    service_port: int
    timeout: int
    temperature: float
```

配置文件:
- `configuration/base.toml`: 基础配置
- `configuration/development.toml`: 开发环境覆盖
- `configuration/production.toml`: 生产环境覆盖

---

### 2. 日志管理

使用 **Loguru** 进行日志管理，支持文件轮转和保留策略。

```python
# core/logger/manager.py
class LogManager:
    def setup(self):
        logger.add(
            self.log_conf.dir + "/app.log",
            rotation=self.log_conf.rotation,
            retention=self.log_conf.retention,
            level=self.log_conf.level
        )
```

---

### 3. 数据库层

#### 3.1 Oracle 元数据 (SQLAlchemy)

使用 SQLAlchemy 异步引擎连接 Oracle 数据库，存储元数据。

```python
# core/database/meta_oracle.py
async_engine = create_async_engine(
    "oracle+oracledb://...",
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True
)

async with get_session() as session:
    result = await session.execute(query)
```

#### 3.2 Oracle 向量存储 (oracledb)

使用 oracledb 直接连接进行向量操作。

```python
# core/database/vec_oracle_pool.py
class AsyncOracleConnectionPoolManager:
    async def get_pool(self, conn_params):
        return await oracledb.create_pool_async(
            user=conn_params.user,
            password=conn_params.password,
            dsn=conn_params.dsn
        )
```

#### 3.3 Elasticsearch 向量存储

用于可选的向量存储方案。

```python
# core/database/vec_elasticsearch.py
class ESClientManager:
    async def get_client(self, connstr: dict):
        return AsyncElasticsearch(connstr)
```

---

### 4. 认证授权

支持三种认证方式:

```python
# core/auth/auth_service.py

# 1. JWT Token 认证
class JWTService:
    def create_access_token(self, data: dict) -> str:
        return jwt.encode(data, self.secret_key, algorithm="HS256")

# 2. API Key 认证
class APIKeyService:
    def generate_api_key(self) -> str:
        return API_KEY_PREFIX + secrets.token_urlsafe(32)

# 3. 密码认证
class PasswordService:
    def verify_password(self, plain: str, hashed: str) -> bool:
        return pwd_context.verify(plain, hashed)
```

---

### 5. 数据访问层 (DAO)

采用 **Entity-Repository** 模式:

```python
# dao/entities/kbot_md_kb.py
class KbotMdKb(Base):
    __tablename__ = "kbot_md_kb"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))

# dao/repositories/kbot_md_kb_repo.py
class KbotMdKbRepository:
    async def get_by_id(self, kb_id: int) -> KbotMdKb | None:
        async with get_session() as session:
            return await session.get(KbotMdKb, kb_id)
```

---

### 6. 微服务层

每个微服务都是独立的 FastAPI 应用，可单独部署:

#### 6.1 LLM 微服务 (端口 18092)

```python
# microservices/llm/llm_service.py
class LLMService:
    async def chat(self, model_name: str, messages: list, **kwargs):
        model = await self.get_llm_model(model_name)
        return await model.chat(messages, **kwargs)
```

支持的 LLM 提供商:
- **OCI Generative AI**
- **OpenAI Compatible APIs**

#### 6.2 Embedding 微服务 (端口 18091)

```python
# microservices/embedding/embed_service.py
class EmbeddingService:
    async def embed_texts(self, model_name: str, texts: list[str]):
        model = await self.get_embedding_model(model_name)
        return await model.embed(texts)
```

支持的 Embedding 提供商:
- **BGE Local** (bge-small-en-v1.5, bge-base-zh-v1.5, qwen3)
- **Cohere API**
- **OpenAI Compatible APIs**

#### 6.3 Reranker 微服务 (端口 18093)

```python
# microservices/reranker/reranker_service.py
class RerankerService:
    async def rerank(self, model_name: str, query: str, docs: list[str]):
        model = await self.get_model(model_name)
        return await model.rerank(query, docs)
```

#### 6.4 VLM 微服务 (端口 18094)

```python
# microservices/vlm/vlm_service.py
class VLMService:
    async def chat(self, model_name: str, image: str, prompt: str):
        model = await self.get_model(model_name)
        return await model.chat(image, prompt)
```

#### 6.5 Parser 微服务 (端口 18095)

文档解析服务，支持多种文档格式。

---

### 7. 业务服务层

#### 7.1 Agent 智能体

##### 传统 Agent

```python
# services/chat/agent_chat.py
class Agent:
    async def run(self, question: str, context: dict):
        # 1. 改写问题
        # 2. 调用知识库搜索
        # 3. 重排序结果
        # 4. 生成答案
```

##### MCP Agent (Model Context Protocol)

```python
# services/chat/mcp_chat.py
class Agent:
    def register_tools(self, tools: list[MCPTool]):
        self.tool_registry.register(tool)

    async def run(self, question: str):
        # LLM 选择工具 -> 执行工具 -> 生成答案
```

#### 7.2 知识库搜索

```python
# services/search/kb_search.py
class KBSearch:
    async def search(self, vector_q: str, fulltext_q: str, security: int):
        # 1. 向量搜索 (Oracle/Elasticsearch)
        # 2. 全文搜索
        # 3. 混合排序
```

#### 7.3 文件解析

```python
# services/dataparse/file_processor.py
class FileProcessor:
    async def process(self, file_path: str, config: dict):
        # 1. 识别文件类型
        # 2. 调用对应解析器
        # 3. 生成文本/Markdown
        # 4. 分块处理
```

支持的格式:
- PDF (pdfplumber, pdfminer.six)
- Office (python-docx, python-pptx, openpyxl)
- HTML/Markdown (BeautifulSoup, markdown)
- 文本 (txt)

---

### 8. MCP 工具

用于 MCP Agent 的工具集:

```python
# mcp_tools/kb_search_tool.py
class KBSearchTool(MCPTool):
    async def execute(self, params: KBSearchToolParams):
        search = KBSearch(params)
        return await search.search(...)
```

可用工具:
- `KBSearchTool`: 知识库搜索
- `InternetSearchTool`: 网络搜索
- `CalculatorTool`: 计算器

---

### 9. 工具类

#### 9.1 模型客户端

```python
# utils/model_client.py
class CallModel:
    async def call_embedding_model(self, model_id: int, texts: list):
        # 调用 Embedding 微服务
        url = f"http://{host}:{port}/v1/embeddings"
        response = await aiohttp.post(url, json={...})
```

#### 9.2 线程池工具

```python
# utils/common.py
async def run_in_thread_pool(func, params, workers=5):
    # 在线程池中批量运行任务
```

---

### 10. API 路由

```python
# api/routers/__init__.py
router = APIRouter(prefix="/api")
router.include_router(kb_router)      # /api/kb/...
router.include_router(agent_router)   # /api/agent/...
router.include_router(model_router)  # /api/model/...
router.include_router(auth_router)   # /api/auth/...
router.include_router(log_router)    # /api/log/...
```

---

## 数据流示例

### 1. Agent 聊天流程

```
用户提问
  ↓
FastAPI Router (/api/agent/chat)
  ↓
Controller (agent_controller.py)
  ↓
Agent (services/chat/agent_chat.py)
  ├─→ 问题改写 (LLM)
  ├─→ 知识库搜索 (KBSearch)
  │   ├─→ 向量搜索 (Oracle/Elasticsearch)
  │   ├─→ 全文搜索
  │   └─→ 重排序 (Reranker 微服务)
  └─→ 生成答案 (LLM)
  ↓
返回结果
```

### 2. 文件上传处理流程

```
用户上传文件
  ↓
FastAPI Router (/api/kb/upload)
  ↓
Controller (kb_controller.py)
  ↓
KB File Operator (services/kb/kb_file_operator.py)
  ├─→ 保存文件
  ├─→ 提取元数据
  └─→ 提交到解析队列
  ↓
File Parser Service (services/dataparse/file_parser_service.py)
  ├─→ 调用 Parser 微服务
  ├─→ 文本分块
  └─→ 生成 Embedding (Embedding 微服务)
  ↓
存储到 Oracle/Elasticsearch
  ↓
完成
```

---

## 部署模式

### 单机部署

所有服务运行在同一进程:

```bash
python kbot_main.py
```

### 微服务部署

每个微服务独立运行:

```bash
# Embedding 服务
python kbot_app_embedding.py

# LLM 服务
python kbot_app_llm.py

# Reranker 服务
python kbot_app_reranker.py

# VLM 服务
python kbot_app_vlm.py

# Parser 服务
python kbot_app_parser.py

# 主服务
python kbot_main.py
```

---

## 技术栈

| 组件 | 技术 |
|------|------|
| Web 框架 | FastAPI |
| ORM | SQLAlchemy 2.0 |
| 数据库 | Oracle 23c |
| 向量存储 | Oracle 23c / Elasticsearch |
| 日志 | Loguru |
| 配置 | Dynaconf |
| 认证 | JWT, bcrypt, python-jose |
| 文档解析 | pdfplumber, python-docx, BeautifulSoup |
| LLM | OCI Generative AI, OpenAI Compatible |
| Embedding | BGE, Cohere, OpenAI |
| 异步 | asyncio, aiohttp |

---

## 环境变量

```bash
# 服务配置
KBOT_SERVICE_NAME=main
KBOT_HOST=0.0.0.0
KBOT_PORT=18099

# Oracle 配置
ORACLE_USERNAME=kbot
ORACLE_PASSWORD=***
ORACLE_HOST=localhost
ORACLE_PORT=1521
ORACLE_SERVICE_NAME=kbotdev
```

---

## 启动流程

1. **加载环境变量** (`load_dotenv`)
2. **初始化配置** (`get_app_config`)
3. **初始化日志** (`LogManager`)
4. **创建 FastAPI 应用** (`FastAPIOffline`)
5. **注册中间件** (CORS, 日志)
6. **注册路由**
7. **启动解析服务管理器** (`FileParserManager`)
8. **启动 Uvicorn 服务器**
