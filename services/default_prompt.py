import string
from loguru import logger
from typing import Any
from services.prompt_service import PromptService

class PromptManager(string.Formatter):
    def __init__(self):
        super().__init__()
        self.prompt_service = PromptService()
        # 建立 默认提示词名称 与 默认内容 的映射关系
        self._default_prompts = {
            "SYSTEM/image2text": DEFAULT_DESCRIBE_PIC_PROMPT,
            "SYSTEM/rewrite_question": DEFAULT_REWRITE_PROMPT,
            "SYSTEM/refresh_summary": DEFAULT_SUMMARY_PROMPT,
            "SYSTEM/rag_final_render": DEFAULT_FINAL_RAG_PROMPT,
            "SYSTEM/user_profile": DEFAULT_USER_PROFILE_PROMPT,
        }

    def get_value(self, key: Any, args: Any, kwargs: Any) -> Any:
        """继承自 string.Formatter: 如果 key 缺失，返回原样 {key}"""
        if isinstance(key, str):
            # 如果 kwargs 中没有该 key，返回 {key} 字符串
            return kwargs.get(key, "{" + key + "}")
        return super().get_value(key, args, kwargs)

    async def generate(self, prompt_name: str, **kwargs) -> str:
        """
        核心方法：DB -> 内存默认 -> 格式化填充
        :param prompt_name: 提示词在系统中的唯一标识 (如 "SYSTEM/user_profile")
        :param kwargs: 需要填充到提示词模板中的变量
        """
        # 1. 获取默认兜底内容
        fallback_content = self._default_prompts.get(prompt_name, "")
        
        # 2. 尝试从数据库获取
        template = fallback_content
        try:
            db_prompt = await self.prompt_service.get_prompt_by_unique_name(prompt_name)
            if db_prompt:
                template = db_prompt
            elif not fallback_content:
                logger.error(f"Prompt '{prompt_name}' not found in DB or Memory.")
                return "" # 或者抛出异常
            else:
                logger.warning(f"Prompt '{prompt_name}' not found in DB, using fallback.")
        except Exception as e:
            logger.warning(f"Failed to fetch prompt '{prompt_name}' from DB: {e}")

        # 3. 使用 self (LazyFormatter) 进行格式化填充
        try:
            return self.format(template, **kwargs)
        except Exception as e:
            logger.error(f"Format prompt '{prompt_name}' failed: {e}")
            return template

# default prompt for rewrite query
DEFAULT_REWRITE_PROMPT = """
你是 RAG 系统的**上下文与身份引擎**。
你的任务是在用户简短输入与深度知识库之间建立桥梁，同时保持记忆干净、无污染。

### 近期对话（短期记忆）
{chat_history}

### 上下文知识
- **历史摘要**: {summary} 
- **当前会话状态**: {session_state}（当前环境、任务专属路径、活跃错误）

### 任务1：查询改写（依赖逻辑）
- **如果 context_relevance >= 0.5**: 执行完整实体注入，使用历史摘要补全缺失的主体或环境信息。
- **如果 context_relevance < 0.5（话题切换）**: 不使用历史摘要中的任何技术细节，将「用户输入」视为全新开始，仅在当前轮次内尽可能做指代消解（如“它”“那个”）。
- 最终生成的 standalone_query 必须干净，不包含上一轮话题的专业术语。

### 任务2：查询改写
- **独立查询**: 重构为自包含的完整查询。
- **关键要求**: 如果 context_relevance 较低，**不要**将摘要中的实体注入到 standalone_query 中。
- **关键词抽取**: 3–5个高密度技术术语，用于 Elasticsearch/Oracle Text 检索。

### 任务3：身份信息 vs 临时状态
- **本轮实体**: 临时上下文（如 PID、错误码、具体日志片段、临时变量）。
- **用户画像更新**: 永久用户特征（如技术水平、职业角色、硬件配置、偏好技术栈）。**不要**把一次性错误放在这里。

### 输出格式（严格JSON）
{{
  "thought": "关于话题切换与实体分类的简要推理",
  "context_relevance": 0.0 到 1.0,
  "standalone_query": "string",
  "search_keywords": ["k1", "k2"],
  "turn_entities": {{}},
  "user_profile_updates": {{}},
  "intent": "troubleshooting | architecture | general"
}}

用户输入：{query}
"""

# default prompt for context summary and user profile summary
DEFAULT_SUMMARY_PROMPT = """
分析以下对话，并以结构化 JSON 格式提供两类摘要。
任务：
1. context_summary：当前会话的技术摘要。重点描述核心问题、环境信息（如操作系统、版本号）以及已验证的解决方案或步骤。
2. profile_summary：对用户的定性描述。根据用户的提问内容与技术深度，识别其职业角色、专业水平和沟通风格。
对话历史：
{history_text}

### 输出格式 (严格遵守 JSON 格式):
{{
  "context_summary": "string",
  "profile_summary": "string"
}}
"""

# default prompt for final rag
DEFAULT_FINAL_RAG_PROMPT = """{system_prompt}

### 当前环境约束 (Session State)
**[必须遵守]** 以下是当前用户的运行环境：
- {env_str}

### 对话背景摘要 (Context Summary)
**[重要上下文]** 本次会话之前的进展：
{context_summary}

### 历史相关经验 (Long-term Memory)
**[仅供参考]** 以下是过往类似场景的经验（带 ⭐ 为用户认可的方案）：
{long_term_memory}

### 核心知识库依据 (Knowledge Base)
**[主要依据]** 请根据以下权威文档回答问题：
{kb_context}

---
请综合上述背景，优先依据【核心知识库】和【环境约束】。

用户当前的问题：{user_question}
助手回答：
"""

# default prompt for user profile
DEFAULT_USER_PROFILE_PROMPT = """
你是一位资深的系统架构师与用户画像专家。请分析对话并输出 JSON 格式的更新记录。

### 原有画像摘要:
{old_summary}

### 最新对话片段:
Q: {question}
A: {answer}

### 任务指令:
1. 分析最新对话，提取用户的专业身份(如DevOps)、使用的技术栈(如Oracle Linux 8)、当前关注的具体项目或痛点。仅从用户提问和对应回答中提取信息，不得引用本提示词中的示例内容（如 DevOps、Oracle Linux 8 等）。
2. 将新提取的信息与原有摘要进行逻辑合并。
3. 如果信息重复，则保留；如果信息冲突（如用户从 Ubuntu 换到了 RHEL），以最新对话为准。
4. 保持摘要简洁、专业，总字数不超过 300 字。
5. 直接输出更新后的摘要文本，不要包含“根据对话...”等废话。
### 额外任务：
6. 请为本次对话生成一个【记忆快照】（Memory Snapshot），用于语义搜索。
### 任务要求:
1. 更新【profile_summary】：合并新老信息，字数<300，专业简洁。如果原有摘要为默认初始化信息，请直接以本次对话内容开启新摘要。
2. 生成【memory_snapshot】：本次对话的核心事实快照，需消解指代（如“它”->“Oracle 26ai”），去除废话。

### 输出格式 (必须为纯 JSON):
{{
    "profile_summary": "更新后的完整画像摘要...",
    "memory_snapshot": "本次对话的高纯度事实快照..."
}}
"""


DEFAULT_DESCRIBE_PIC_PROMPT = """
你是一个专业的 RAG 系统图像解析助手。请根据提供的图片和文档上下文进行分析。

任务目标：
将图片内容转化为有助于搜索和问答的纯文本描述。

遵循准则：

静默原则：如果图片是背景图、logo、点缀性边框、或者是没有任何实际业务含义的装饰性图片，请直接返回“[NONE]”。

信息优先级：优先提取图中的文字（OCR）、图表标题、坐标轴标签、流程图节点名称。

极简描述：禁止使用“精美的”、“科幻的”、“令人惊叹的”等修辞词。直接描述核心主体，例如：“[流程图] 展示了晶圆清洗的三个步骤：酸洗、水洗、干燥”。

禁止幻觉：只描述你确信看到的。如果图片模糊无法辨认，请直接说明“图片内容无法辨识”，不要推测。

上下文对齐：参考提供的上下文（{current_header}），如果图片内容与标题无关，请缩减描述。

输出格式：
[图片类型]：[核心信息总结]
"""