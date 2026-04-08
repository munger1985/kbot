import string
from loguru import logger
from services.prompt_service import PromptService

class LazyFormatter(string.Formatter):
    def get_value(self, key, args, kwargs):
        # 如果大括号里的 key 在 kwargs 中找不到，直接原样返回
        if isinstance(key, str):
            return kwargs.get(key, "{" + key + "}")
        return super().get_value(key, args, kwargs)
    
class DefaultPrompt:
    def __init__(self):
        self.prompt_service = PromptService()
    
    async def get_prompt_content(self, unique_name: str, default_content: str) -> str:
        """统一的 Prompt 获取策略：DB -> Default"""
        try:
            prompt = await self.prompt_service.get_prompt_by_unique_name(unique_name)
            if not prompt:
                logger.warning(f"Prompt '{unique_name}' not found in DB, using fallback.")
                return default_content
            return prompt
        except Exception as e:
            logger.warning(f"Failed to fetch prompt '{unique_name}' from DB, using fallback. Error: {e}")
        return default_content

# default prompt for rewrite query
DEFAULT_REWRITE_PROMPT = """
You are the Context and Identity Engine for a High-Performance RAG system.
Your mission is to bridge the gap between user's brief input and the deep knowledge base while maintaining a clean, non-contaminated memory.

### Context Knowledge
- **Historical Summary**: {summary} 
- **Active Session State**: {session_state} (Current environment, task-specific paths, active errors)

### Task 1: Query Rewriting (Dependency Logic)
- **IF context_relevance >= 0.5**: Perform full entity injection. Use the Historical Summary to fill in missing subjects or environment details.
- **IF context_relevance < 0.5 (Topic Shift)**: DO NOT use any technical details from the Historical Summary. Treat the 'User Input' as a fresh start, only performing coreference resolution (like 'it' or 'that') within the current turn if possible. 
- The resulting 'standalone_query' must be clean and free from the previous topic's jargon.

### Task 2: Query Rewriting
- **Standalone Query**: Reconstruct the query to be self-contained. 
- **CRITICAL**: If "context_relevance" is low, do NOT inject entities from the Summary into the standalone_query. 
- **Keyword Extraction**: 3-5 high-density technical terms for Elasticsearch/Oracle Text.

### Task 3: Identity vs. Transient State
- **Turn Entities**: Temporary context (e.g., PIDs, error codes, specific log snippets, temporary variables).
- **User Profile Updates**: Permanent user traits (e.g., Skill level, Job role, Hardware specs, preferred Stack). Do not put one-off errors here.

### Output Format (Strict JSON)
{{
  "thought": "Brief reasoning about topic shift and entity classification",
  "context_relevance": 0.0 to 1.0,
  "standalone_query": "string",
  "search_keywords": ["k1", "k2"],
  "turn_entities": {{}},
  "user_profile_updates": {{}},
  "intent": "troubleshooting | architecture | general"
}}

User Input: {query}
"""

# default prompt for context summary and user profile summary
DEFAULT_SUMMARY_PROMPT = """
Analyze the following conversation and provide two types of summaries in a structured JSON format.

### Tasks:
1. **context_summary**: A technical summary of the current session. Focus on the core problem, environment (e.g., OS, versions), and verified solutions/steps.
2. **profile_summary**: A qualitative description of the user. Identify their professional role, expertise level, and communication style based on their questions and technical depth.

### Dialogue History:
{history_text}

### Output Format (Strict JSON):
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