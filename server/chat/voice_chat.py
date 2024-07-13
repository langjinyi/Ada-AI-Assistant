# _*_ coding: utf-8 _*_
# @Time    : 2024/7/10 12:55
# @Author  : JINYI LIANG

from fastapi import Body
from configs import LLM_MODELS, TEMPERATURE, PROMPT_TEMPLATES
from langchain.chains import LLMChain
import asyncio
from langchain.prompts.chat import ChatPromptTemplate
from typing import List, Optional, Union
from langchain.chat_models import ChatOpenAI
from server.utils import get_model_worker_config, fschat_openai_api_address, api_address
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from fastapi import Body, UploadFile, File
from fastapi.responses import FileResponse
import requests

headers = {
    'accept': 'application/json',
}
api_base_url = api_address()

v2t_api = "/v2t/transform"
v2t_url = f"{api_base_url}{v2t_api}"

t2t_api = "/t2t/chat"
t2t_url = f"{api_base_url}{t2t_api}"


async def voice_chat(audio_file: UploadFile = File(..., description="用户输入语音文件")):
    # 读取WAV文件内容
    files = {'audio_file': ('output.wav', open('output.wav', 'rb'), 'audio/wav')}
    v2t_response = requests.post(v2t_url, headers=headers, files=files)
    print(v2t_response.text)
    # data = {
    #     "query": v2t_response.text
    # }
    # t2t_response = requests.post(t2t_url, headers=headers, json=data)

    return v2t_response.text
