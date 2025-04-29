from fastapi import Body
from typing import Annotated
from typing import Literal, Iterable
import tiktoken
import numpy as np
import time
import shortuuid
from pydantic import BaseModel, Field
import base64
from util import BaseResponse
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import uvicorn
from typing import Literal, Optional, List, Dict, Any, Union
import embedding_models,llm_models


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


# 创建一个简单的提示模板
prompt_template = PromptTemplate(
    input_variables=["question"],
    template="请回答下面的问题: {question}"
)


# # 为每个LLM创建一个chain
# openai_chain = LLMChain(llm=openai_llm, prompt=prompt_template)
# huggingface_chain = LLMChain(llm=huggingface_llm, prompt=prompt_template)
class ChatCompletionRequest(BaseModel):
    model: str
    messages: Union[str, List[Dict[str, str]]] = [{
        "role": "user",
        "content": "Hello!"
    }]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = -1
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None


class CompletionRequest(BaseModel):
    model: str
    messages: str
    max_tokens: int = 100
    temperature: float = 0.3


chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a helpful and smart ai called kbot"),
        ("human", "{input}"),

    ]
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


class EmbeddingsRequest(BaseModel):
    input: str | list[str] | Iterable[int | Iterable[int]]
    model: str
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: int | None = None  # not used.
    user: str | None = None  # not used.


class Embedding(BaseModel):
    object: Literal["embedding"] = "embedding"
    embedding: list[float] | bytes
    index: int


class EmbeddingsUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[Embedding]
    model: str
    usage: EmbeddingsUsage


ENCODER = tiktoken.get_encoding("cl100k_base")


def extendApp(app):
    @app.post("/v1/chat/completions", tags=["OpenAI API compatible"],   summary="chat with OpenAI API")
    def create_completion(request: Annotated[ChatCompletionRequest,
    Body(
        examples=[
            {
                "model": "OCI-meta.llama-3.1-405b-instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello!"
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 0
            }
        ],
    ),
    ]):
        try:
            query_llm = LLMChain(llm=llm_models.MODEL_DICT.get(
                request.model), prompt=chat_template)
            logger.info(
                f"#### chat with LLM {request.model}: ", llm_models.MODEL_DICT.get(request.model))
            logger.info(
                f"#### user: {request.messages[0].get('content')} ")
            response = query_llm.invoke(
                {"input": request.messages[0].get('content')})
            usage = UsageInfo()
            choices = []

            choices.append(
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant", content=response["text"]),
                    finish_reason="stop",
                )
            )
            # if "usage" in content:
            #     task_usage = UsageInfo.parse_obj(content["usage"])
            #     for usage_key, usage_value in task_usage.dict().items():
            #         setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)
            return ChatCompletionResponse(model=request.model, choices=choices, usage=usage)

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    @app.get("/v1/models", tags=["OpenAI API compatible"],   summary="get openai api format models")
    def get_models( ):
        try:
            modelNames= llm_models.MODEL_DICT.keys()

            result = {
                        "object": "list",
                        "data": [
                            {
                                "id": name,
                                "object": "model",
                                "created": 1677649429,
                                "owned_by": "kbot",
                                "permission": []
                            }
                            for name in modelNames
                        ]
                    }

# 将结果转换成格式化的 JSON 字符串并打印
# print(json.dumps(result, indent=2, ensure_ascii=False))
            return BaseResponse(data= result)

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/embeddings", tags=["OpenAI API compatible"],   summary="embeddings with  OpenAI API")
    def embed(embeddings_request: Annotated[
        EmbeddingsRequest,
        Body(
            examples=[
                {
                    "model": "OCI-cohere.embed-multilingual-v3.0",
                    "input": [
                        "Your text string goes here"
                    ],
                }
            ],
        ),
    ]):
        data = []
        texts = None
        if isinstance(embeddings_request.input, str):
            texts = [embeddings_request.input]
        elif isinstance(embeddings_request.input, list):
            texts = embeddings_request.input
        elif isinstance(embeddings_request.input, Iterable):
            encodings = []
            for inner in embeddings_request.input:
                if isinstance(inner, int):
                    encodings.append(inner)
                else:
                    text = ENCODER.decode(list(inner))
                    texts.append(text)
            if encodings:
                texts.append(ENCODER.decode(encodings))
        embeddingModel = embedding_models.EMBEDDING_DICT.get(embeddings_request.model)
        embeddings = embeddingModel.embed_documents(texts)
        for i, embedding in enumerate(embeddings):
            if embeddings_request.encoding_format == "base64":
                arr = np.array(embedding, dtype=np.float32)
                arr_bytes = arr.tobytes()
                encoded_embedding = base64.b64encode(arr_bytes)
                data.append(Embedding(index=i, embedding=encoded_embedding))
            else:
                data.append(Embedding(index=i, embedding=embedding))
        response = EmbeddingsResponse(
            data=data,
            model=embeddings_request.model,
            usage=EmbeddingsUsage(
                prompt_tokens=1,
                total_tokens=0 + 1,
            ),
        )
        return response

    return app
