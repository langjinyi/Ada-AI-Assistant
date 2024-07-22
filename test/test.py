# _*_ coding: utf-8 _*_
# @Time    : 2024/7/16 18:53
# @Author  : JINYI LIANG
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate

from configs import PROMPT_TEMPLATES

llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key="",
    openai_api_base="https://api.deepseek.com/v1",
    temperature=1,
    max_tokens=20)

chat_prompt = ChatPromptTemplate.from_messages(
    [("system", PROMPT_TEMPLATES['girl_friend(chat)']),
     ("human", "{input}")],
)

chain = LLMChain(prompt=chat_prompt, llm=llm, verbose=True)
print(chain("你是谁"))
