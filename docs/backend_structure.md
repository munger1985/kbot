## 后端框架结构

```bash
backend/
├── api/
│   ├── controllers/           # API控制器
│   │   └── health.py
│   ├── routers/               # API路由
│   │   └── __init__.py
│   ├── schemas/               # Pydantic响应模型
│   └── routers.py             # FastAPI路由注册
├── core/
│   ├── config/                # 配置管理
│   │   └── __init__.py
│   ├── database/              # 数据库连接
│   │   └── oracle.py          # Oracle数据库连接
│   ├── exceptions/            # 全局异常处理
│   └── log/                   # 日志管理
│       ├── __init__.py
│       └── logger.py
├── dao/                       # 数据访问层
│   ├── entities/              # 实体定义
│   │   └── base.py
│   └── repositories/          # 数据仓库实现
├── logs/                      # 日志文件
│   └── app.log
├── models/                    # 模型定义
│   ├── embedding/             # 嵌入模型
│   ├── llm/                   # 大语言模型
│   ├── reranker/              # 重排序模型
│   └── vlm/                   # 视觉语言模型
├── services/                  # 业务服务层
│   ├── chat/                  # 聊天服务
│   ├── dataparse/             # 数据解析服务
│   ├── knowbase/              # 知识库服务
│   ├── ocikbot/               # Oracle Kbot服务
│   └── search/                # 搜索服务
├── utils/                     # 工具类
├── main.py                    # FastAPI启动入口
├── requirements.txt           # 依赖文件
├── settings.toml              # 配置文件
└── start.sh                   # 启动脚本
```

## 核心组件说明

### 1. 配置管理

使用`settings.toml`文件管理配置，通过`core/config/__init__.py`提供统一访问接口。

```python
# core/config/__init__.py
import toml
from pathlib import Path

config_path = Path(__file__).parent.parent.parent / "settings.toml"
config = toml.load(config_path)
```

### 2. 日志管理

使用loguru进行日志管理，配置在`core/log/logger.py`中。

```python
# core/log/logger.py
from loguru import logger
import sys

logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("logs/app.log", rotation="100 MB", retention="10 days")
```

### 3. 数据库操作

使用SQLAlchemy进行Oracle数据库操作，基础配置在`core/database/oracle.py`中。

```python
# core/database/oracle.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

engine = create_engine("oracle://user:pass@host:port/sid")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
```

### 4. 数据访问层

数据访问层分为实体定义(`dao/entities/`)和仓库实现(`dao/repositories/`)。

```python
# dao/entities/base.py
from sqlalchemy.ext.declarative import declarative_base
from core.database.oracle import Base

class EntityBase(Base):
    __abstract__ = True
    # 公共字段和方法
```

### 5. 服务层架构

服务层按功能划分为多个子服务，每个服务包含完整的业务逻辑：

- `chat/`: 处理聊天相关业务
- `dataparse/`: 数据解析和处理
- `knowbase/`: 知识库管理
- `ocikbot/`: Oracle Kbot OCI特定功能
- `search/`: 搜索功能实现

每个服务应包含完整的业务逻辑，并通过API层暴露接口。