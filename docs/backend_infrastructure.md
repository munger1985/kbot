
## 后端框架

```bash
backend/
├── core/
│   ├── config/                # TOML配置管理
│   │   ├── settings.toml       # 主配置文件
│   │   ├── __init__.py        # 初始化配置加载器
│   │   └── env/               # 环境配置
│   │       ├── production.toml
│   │       └── development.toml
│   ├── log/                   # Loguru日志管理
│   │   └── logger.py          # 全局日志配置
│   ├── database/              # SQLAlchemy核心
│   │   ├── base.py            # ORM基类
│   │   ├── session.py         # 会话管理
│   │   └── redis_client.py    # Redis集成
│   └── exceptions/            # 全局异常处理
├── api/
│   ├── controllers/
│   │   └── ...                
│   └── schemas/               # Pydantic响应模型
│       ├── image_search.py
│       └── ...              
├── services/
│   ├── image_search/
│   │   └── engine.py          # 业务逻辑处理
│   └── ...                     
├── repositories/              # 数据访问层
│   ├── knowledge_repo.py      # SQLAlchemy操作类
│   ├── vector_repo.py         # 向量库操作
│   └── cache_repo.py          # Redis操作
├── models/                    # SQLAlchemy ORM模型
│   ├── knowledge.py           # 知识模型
│   ├── search_log.py          # 日志模型
│   └── __init__.py            # 声明式基类导入
├── engines/
│   ├── image_search/
│   │   ├── tensorflow_engine.py # 模型引擎
│   │   └── requirements.txt    
│   └── ...                  
├── tasks/                     
│   └── celery_app.py          # Celery配置
└── main.py                    # FastAPI启动入口
```

#### 1. 配置管理（toml）

- 使用`toml`文件管理配置，通过`core/config/__init__.py`提供统一访问接口。
- 示例代码
```python
# core/config/__init__.py
import toml
from pathlib import Path

# 根据环境变量加载配置
ENV = os.getenv("ENVIRONMENT", "development")

config_path = Path(__file__).parent / f"environments/{ENV}.toml"
config = toml.load(config_path)
```
#### 2. 日志管理（loguru）

- 在`core/logger/__init__.py`中初始化loguru，并配置日志格式、输出、旋转等。
- 示例代码：
```python
# core/logger/__init__.py
from loguru import logger
import sys
from core.config import config  # 导入配置

# 从配置中获取日志级别和日志文件路径
log_level = config["logging"]["level"]
log_path = config["logging"]["path"]

# 移除默认设置
logger.remove()
# 添加控制台输出
logger.add(sys.stderr, level=log_level)
# 添加文件输出
logger.add(log_path, rotation="100 MB", retention="10 days", level=log_level)

# 供其他模块导入
__all__ = ["logger"]
```
#### 3. 数据库操作（SQLAlchemy）

- 在`models`目录下使用SQLAlchemy定义ORM模型。
    
- 在`repositories/relational_db.py`中使用SQLAlchemy会话进行数据库操作。
    
- 通过`dependencies.py`管理数据库会话的依赖注入。
    
- 模型示例：
```python
# models/knowledge_model.py
from sqlalchemy import Column, String, LargeBinary, Text
from .base import Base  # 假设有一个基础类

class KnowledgeEntry(Base):
    __tablename__ = "knowledge_entries"

    id = Column(String(36), primary_key=True)
    title = Column(String(255))
    content = Column(Text)
    # 注意：向量通常存储在向量数据库中，这里可以存一个引用ID或不用
    # 我们假设向量存储在Milvus/FAISS，所以关系库只存业务数据
    vector_id = Column(String(50))  # 对应向量库中的ID
```

- 依赖注入（在 FastAPI 中）：
```python
# dependencies.py
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

# 从配置中读取数据库URL
from core.config import config

DATABASE_URL = config["database"]["url"]
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 用于注入会话
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()
```

- 在控制器中使用：
```python
from dependencies import get_db
from fastapi import Depends

class ImageSearchController:
    @router.post("/search-by-image")
    async def search_by_image(file: UploadFile, db: Session = Depends(get_db)):
        repo = KnowledgeRepository(db)
        # 使用repo进行查询
```


## 关键特性实现

1. #### 日志全链路追踪
```python
# core/log/middleware.py (FastAPI中间件)
async def log_middleware(request: Request, call_next):
    log.info(f"Request: {request.method} {request.url}")
    
    start_time = time.time()
    response = await call_next(request)
    
    process_time = time.time() - start_time
    log.bind(
        path=request.url.path,
        method=request.method,
        latency=process_time
    ).info("Request completed")
    
    return response
```

2. #### 热配置加载
```python
# core/config/reloader.py
import time
import threading
from . import config

class ConfigReloader:
    def __init__(self, interval=60):
        self.interval = interval
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        
    def _run(self):
        while True:
            time.sleep(self.interval)
            config.refresh()  # 重新加载配置
            log.info("Configuration reloaded")
            
    def start(self):
        self.thread.start()
```