# _*_ coding: utf-8 _*_
# @Time    : 2024/7/9 18:15
# @Author  : JINYI LIANG
from fastapi import Body
from configs import LLM_MODELS, TEMPERATURE, PROMPT_TEMPLATES
from langchain.chains import LLMChain
import asyncio
from langchain.prompts.chat import ChatPromptTemplate
from typing import List, Optional, Union
from langchain.chat_models import ChatOpenAI
from server.utils import get_model_worker_config, fschat_openai_api_address
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import MessagesPlaceholder


async def chat(query: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
               stream: bool = Body(False, description="流式输出"),
               model_name: str = Body(LLM_MODELS, description="LLM 模型名称。"),
               temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=2.0),
               max_tokens: Optional[int] = Body(None, description="限制LLM生成Token数量，默认None代表模型最大值"),
               # top_p: float = Body(TOP_P, description="LLM 核采样。勿与temperature同时设置", gt=0.0, lt=1.0),
               prompt_name: str = Body("default", description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
               ):
    config = get_model_worker_config(model_name)
    model = ChatOpenAI(
        streaming=True,
        verbose=True,
        # callbacks=callbacks,
        openai_api_key=config.get("api_key", "EMPTY"),
        openai_api_base=config.get("api_base_url", fschat_openai_api_address()),
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_proxy=config.get("openai_proxy"),
    )

    chat_prompt = ChatPromptTemplate.from_messages(
        [("system", PROMPT_TEMPLATES['girl_friend']),
         MessagesPlaceholder(variable_name="history"),
         ("human", "{input}")])

    memory = ConversationBufferWindowMemory(
        memory_key="history",  # 此处的占位符可以是自定义
        return_messages=True,
        k=5
    )

    chain = LLMChain(prompt=chat_prompt, llm=model, memory=memory, verbose=True)

    return chain(query)
