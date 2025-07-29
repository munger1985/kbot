import spacy
from transformers.pipelines import pipeline
import sys
from pathlib import Path
# Add both project root and backend directory to Python path
project_root = Path(__file__).resolve().parent.parent.parent.parent
backend_dir = project_root / "backend"
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(project_root))

from backend.core.config import settings

# 指定本地模型路径
model_name = settings["nlp"]["model_name"]
nlp = spacy.load(model_name)  # 直接加载本地模型

# 验证使用
text = "郑和下西洋是什么时候"
doc = nlp(text)

# 提取实体
entities = [(ent.text, ent.label_) for ent in doc.ents]
print(entities)

