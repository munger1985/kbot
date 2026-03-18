from services.search.result import TxtBaseSearchResult

class ContextBuilder:
    @staticmethod
    def build_final_prompt(
        system_prompt: str,
        user_question: str,
        kb_results: list[TxtBaseSearchResult],
        short_term_memory: str = "",
        long_term_memory: str = ""
    ) -> str:
        """
        组装最终的 Prompt，包含长短期记忆和带路径基因的知识库内容
        """
        
        # 1. 格式化知识库检索结果 (注入路径基因以增强上下文理解)
        kb_segments = []
        for i, res in enumerate(kb_results):
            # 拼接章节路径，例如: [手册 > 第三章 > 安装逻辑] 内容...
            path_str = " > ".join(res.path_names) if res.path_names else "通用文档"
            segment = f"[参考资料 {i+1} | 来源: {path_str}]\n{res.content}"
            kb_segments.append(segment)
        
        kb_context = "\n\n".join(kb_segments) if kb_segments else "未找到直接相关的知识库资料。"

        # 2. 构造分层上下文模板
        final_prompt = f"""{system_prompt}

### 记忆背景（Long-term Memory）
以下是你与用户过去在其他会话中讨论过的相关经验（仅供参考）：
{long_term_memory if long_term_memory else "暂无相关历史经验。"}

### 当前会话历史（Short-term Memory）
以下是本次对话的前文，请保持逻辑连贯：
{short_term_memory if short_term_memory else "对话刚开始。"}

### 核心知识依据（Retrieved Context）
请务必根据以下权威资料回答问题。如果资料中未提及，请诚实告知用户：
{kb_context}

---
请根据上述所有背景信息，专业且准确地回答用户的问题。

用户的问题：{user_question}
助手回答："""

        return final_prompt