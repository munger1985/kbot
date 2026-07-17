from typing import AsyncGenerator, Any
from loguru import logger
from skills import BaseSkill
from utils.clients import AIModelClient
from core.dictionary import PacketType
from agent.common import ContextMemory

class ChitChatSkill(BaseSkill):
    """
    Lightweight chat skill: For handling non-business conversations.
    Directly use get_llm_stream_parsed to obtain parsed streaming data.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "ChitChatSkill"
        self.description = "Handle general conversations, greetings and open questions outside RAG business scenarios."
        self.model_client = AIModelClient()

    async def run_stream(
        self, 
        context: ContextMemory,
        **kwargs
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Fully asynchronous streaming call: Rewritten question -> Parsed content package
        """
        # Extract parameters from context
        current_exec = context["current_execution"]
        if isinstance(current_exec, dict):
            task_input = current_exec.get("resolved_input") or current_exec.get("task_description") or ""
        else:
            task_input = current_exec or ""
        task_input = task_input or context["standalone_query"] or context["question"] or "hello"
        model_name = context["llm_model"]

        messages = [
            {"role": "system", "content": f"You are an intelligent assistant. The user's language is: {context.get('user_language', 'English')}. You MUST reply in {context.get('user_language', 'English')}. Please provide concise and accurate answers based on the user's questions."},
            {"role": "user", "content": f"{task_input}\n\n【重要语言指令】用户的语言是 {context.get('user_language', 'English')}。你必须使用 {context.get('user_language', 'English')} 进行回答，严禁使用其他语言。"}
        ]
        logger.info(f"[{self.name}] Trigger chat stream, using model: {model_name}")
        logger.info(f"[{self.name}] user_language from context: {context.get('user_language', 'NOT_SET')!r}")
        logger.info(f"[{self.name}] System prompt: {messages[0]['content'][:200]}")

        try:
            # Call the existing parsing method, automatically handle JSON parsing and field extraction
            async for chunk in self.model_client.get_llm_stream_parsed(
                model_name=model_name,
                prompt=messages,
                **kwargs
            ):
                # Prioritize reasoning stream (Thought/Reasoning)
                if chunk.reasoning_content:
                    yield {
                        "type": PacketType.THOUGHT,
                        "content": chunk.reasoning_content
                    }
                
                # Process standard answer stream content
                if chunk.content:
                    yield {
                        "type": PacketType.ANSWER,
                        "content": chunk.content
                    }
                            
        except Exception as e:
            logger.error(f"[{self.name}] Runtime exception: {str(e)}", exc_info=True)
            content = f"⚠️ There was a problem with the conversation generation, please try again later. \n"
            yield {"type": PacketType.ERROR, "content": content}
