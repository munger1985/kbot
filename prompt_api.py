# -*- coding: UTF-8 -*-
from typing import List
from langchain.output_parsers import pydantic
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String, DateTime, func
from kb_api import Prompt
import config
from kb_api import with_session
from fastapi.responses import Response
from fastapi import Body
from loguru import logger

@with_session
def add_prompt(session,
                   name: str,
                    template:str
                   ):
    p = session.query(Prompt).filter_by(name=name).first()
    if  p :
        return False
    else :
        prompt=Prompt(name=name,template=template)
        session.add(prompt)
    return True


@with_session
def list_prompt_from_db(session):
    prompts = session.query(Prompt.name).filter( ).all()
    ps = [p[0] for p in prompts]
    return ps

@with_session
def update_prompt_from_db(session, name,content):

        # 找到名为John的prompt，并更新他
        session.query(Prompt).filter(Prompt.name == name).update({Prompt.template: content}, synchronize_session=False)
        # 提交更改
        session.commit()
@with_session
def load_prompt_from_db(session, name):
    if name=='rag_default':
        rag_default_prompt_content= '''
你是一个乐于助人、尊重他人、诚实的AI助手。
请参考下面的上下文内容，回答后面的问题。如果您不知道答案，就回答说不知道，不要试图编造答案。
如果要回答的问题是中文，则用中文回答问题。
如果要回答的问题是英文，则用英文回答问题。
下面是上下文：
{context}
下面是要回答的问题：{question}
                                    '''
        return rag_default_prompt_content
    kb = session.query(Prompt).filter_by(name=name).first()
    if kb:
        return kb.template
    else:
        return None
@with_session
def delete_prompt_from_db(session, name):
    prompt = session.query(Prompt).filter_by(name=name).first()
    if prompt:
        session.delete(prompt)
        session.commit()
    else:
        return None
from fastapi import  Form
from pydantic import BaseModel
import pydantic

class PromptResponse(BaseModel):
    data: str = pydantic.Field(..., description="data returned from vector store")
    status: str = pydantic.Field(..., description="Response text")
    err_msg: str = pydantic.Field(..., description="Response text")

    class Config:
        json_schema_extra = {
            "example": {
                "data": [{"content": "xxx", "score": 1, "source": "llm"},
                         {"content": "yyy", "score": 0.78, "source": "source file url"}],
                "status": "success",
                "err_msg": ""
            }
        }
from util import  BaseResponse
class ListResponse(BaseResponse):
    data: List[str] = pydantic.Field(..., description="List of knowledge base names")

    class Config:
        json_schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
                "data": ["bank", "medical", "OCI info"],
            }
        }

def list_prompts():
    arr=list_prompt_from_db()
    arr.append('default')
    arr.append('rag_default')

    return ListResponse(data=list(arr))



def get_prompt(   name: str = Form(..., description="prompt name", examples=["llama2Prompt"]),
                ) -> Response:
    kk= load_prompt_from_db(name)

    if  kk:
        return Response(content=kk, media_type="text/plain")
    else:
        return  Response(content='{query}', media_type="text/plain")

def delete_prompt(
             name: str = Form(..., description="prompt name", examples=["llama2Prompt"]),
              ):
    logger.info(f"##delete_prompt name:{name}")
    delete_prompt_from_db(name)
    return PromptResponse(data=f"deleted prompt {name} ", status="ok",err_msg="")

def create_prompt(
            name: str = Form(..., description="prompt name", examples=["llama2Prompt"]),
            template: str = Form(..., description="when you chat with llm, {query} is variable, chat with rag, {query} {context} are variables", examples=["you are an AI, answer my question {query}"]),
              ) -> Response:
    logger.info(f"##name:{name} template:{template}")
    kk= load_prompt_from_db(name)
    logger.info(f"##kk:{kk} ")
    if  kk:
        return PromptResponse(data=f"the prompt existed, change another name ", status="ok",err_msg="change another name")
    else:
        add_prompt(name,template)
        return PromptResponse(data=f"created prompt {name} ", status="ok",err_msg="")



def update_prompt(
              name: str = Form(..., description="prompt name", examples=["llama2Prompt"]),
            template: str = Form(..., description="when you chat with llm, {query} is variable, chat with rag, {query} {context} are variables", examples=["you are an AI, answer my question {query}"]),
             ) -> PromptResponse:

    update_prompt_from_db(name,template)
    return PromptResponse(status="ok" , data=f"updated prompt {name} ",err_msg="")




