# _*_ coding: utf-8 _*_
# @Time    : 2024/7/6 21:02
# @Author  : JINYI LIANG
import io
import json
import tempfile
import wave
import zipfile
from ast import literal_eval
from typing import Optional, Union, List, Dict, Annotated

import librosa
import nltk
import sys
import os

import numpy as np
import torch
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from configs.model_config import NLTK_DATA_PATH, V2T_MODELS, T2V_MODELS, MODEL_PATH
from configs.server_config import OPEN_CROSS_DOMAIN
import argparse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
import ChatTTS
from server.utils import BaseResponse, FastAPI, get_model_worker_config, fschat_openai_api_address, \
    remove_special_characters, add_wavs_to_zip
from fastapi import Body, UploadFile, File, Form
from fastapi.responses import FileResponse, StreamingResponse
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from fastapi import Body
from configs import LLM_MODELS, TEMPERATURE, PROMPT_TEMPLATES
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
import soundfile as sf

nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path


class History(BaseModel):
    role: str
    content: str


async def document():
    return RedirectResponse(url="/docs")


def create_app(run_mode: str = None):
    app = FastAPI(
        title="Ada API Server",
    )
    # Add CORS middleware to allow all origins
    # 在config.py中设置OPEN_DOMAIN=True，允许跨域
    # set OPEN_DOMAIN=True in config.py to allow cross-domain
    if OPEN_CROSS_DOMAIN:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    mount_app_routes(app, run_mode=run_mode)
    return app


def mount_app_routes(app: FastAPI, run_mode: str = None):
    # load model and processor
    v2t_processor = WhisperProcessor.from_pretrained(MODEL_PATH["v2t_model"][V2T_MODELS])
    v2t_model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH["v2t_model"][V2T_MODELS]).to("cuda")
    v2t_model.config.forced_decoder_ids = None

    tts = ChatTTS.Chat()
    # compile=True效果好，但是更慢
    tts.load(device="cuda", source='custom', custom_path=MODEL_PATH["t2v_model"][T2V_MODELS], compile=False)

    app.get("/",
            response_model=BaseResponse,
            summary="swagger 文档")(document)

    # Tag: T2T
    @app.post("/t2t/chat",
              tags=["Chat"],
              summary="与llm模型text to text对话(通过LLMChain)",
              )
    async def chat(query: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
                   history: List[Dict] = Body([],
                                              description="历史对话，设为一个整数可以从数据库中读取历史消息",
                                              examples=[[
                                                  {"role": "user",
                                                   "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                                  {"role": "assistant", "content": "虎头虎脑"}]]
                                              ),
                   K: int = Body(3, description="保留K轮历史"),
                   stream: bool = Body(False, description="流式输出"),
                   model_name: str = Body(LLM_MODELS, description="LLM 模型名称。"),
                   temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=2.0),
                   max_tokens: Optional[int] = Body(None, description="限制LLM生成Token数量，默认None代表模型最大值"),
                   # top_p: float = Body(TOP_P, description="LLM 核采样。勿与temperature同时设置", gt=0.0, lt=1.0),
                   prompt_name: str = Body("default",
                                           description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
                   ):
        config = get_model_worker_config(model_name)
        model = ChatOpenAI(
            streaming=stream,
            verbose=True,
            # callbacks=callbacks,
            openai_api_key=config.get("api_key", "EMPTY"),
            openai_api_base=config.get("api_base_url", fschat_openai_api_address()),
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_proxy=config.get("openai_proxy"),
        )
        converted_examples = []
        for his in history:
            converted_examples.append(f'{his["role"]}：{his["content"]}')
        converted_output = "\n".join(converted_examples[-K:])

        chat_prompt = ChatPromptTemplate.from_messages(
            [("system", PROMPT_TEMPLATES['girl_friend'] + converted_output),
             ("human", "{input}")])

        chain = LLMChain(prompt=chat_prompt, llm=model, verbose=True)

        return chain(query)

    # Tag: voice chat
    @app.post("/voice/chat",
              tags=["Voice Chat"],
              summary="与llm模型语音对话(通过LLMChain)",
              )
    async def voice_chat(audio_file: UploadFile = File(..., description="用户输入语音文件"),
                         history: List[str] = Form([], description="用户聊天记录"),
                         K: int = Form(3, description="保留K轮历史"),
                         stream: bool = Form(False, description="流式输出"),
                         model_name: str = Form(LLM_MODELS, description="LLM 模型名称。"),
                         temperature: float = Form(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=2.0),
                         max_tokens: Optional[int] = Form(None,
                                                          description="限制LLM生成Token数量，默认None代表模型最大值"),
                         # top_p: float = Form(TOP_P, description="LLM 核采样。勿与temperature同时设置", gt=0.0, lt=1.0),
                         prompt_name: str = Form("default",
                                                 description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
                         speed: Optional[str] = Form("[speed_2]", description="语速"),
                         seed: Optional[int] = Form(300, description="音色")
                         ):

        # history = json.loads(f"{history}")
        history = [literal_eval(item) for item in history]

        # Read the UploadFile into a file-like object
        audio_bytes = await audio_file.read()
        audio_io = io.BytesIO(audio_bytes)

        audio, sampling_rate = librosa.load(audio_io, sr=16000)

        input_features = v2t_processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to("cuda")
        prompt_ids = v2t_processor.get_prompt_ids("以下是普通话的句子:", return_tensors="pt").to("cuda")

        predicted_ids = v2t_model.generate(input_features=input_features, prompt_ids=prompt_ids)
        transcription = v2t_processor.batch_decode(predicted_ids, skip_special_tokens=True)

        query = transcription[0][len(prompt_ids) - 1:]

        config = get_model_worker_config(model_name)
        model = ChatOpenAI(
            streaming=stream,
            verbose=True,
            # callbacks=callbacks,
            openai_api_key=config.get("api_key", "EMPTY"),
            openai_api_base=config.get("api_base_url", fschat_openai_api_address()),
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_proxy=config.get("openai_proxy"),
        )

        converted_examples = []
        if len(history) >= 2:
            for his in history:
                converted_examples.append(f'{his["role"]}：{his["content"]}')
        converted_output = "\n".join(converted_examples[-K * 2:])

        chat_prompt = ChatPromptTemplate.from_messages(
            [("system", PROMPT_TEMPLATES['girl_friend'] + converted_output),
             ("human", "{input}")])

        chain = LLMChain(prompt=chat_prompt, llm=model, verbose=True)

        text = chain(query)
        text_ = text["text"]
        text_ = remove_special_characters(text_)

        # std, mean = chat.sample_random_speaker
        torch.manual_seed(seed)
        rand_spk = tts.sample_random_speaker()

        params_infer_code = ChatTTS.Chat.InferCodeParams(
            prompt=speed,
            spk_emb=rand_spk,  # add sampled speaker
            temperature=.3,  # using custom temperature
        )

        params_refine_text = ChatTTS.Chat.RefineTextParams(
            prompt='[oral_2][laugh_0][break_6]',
        )

        wavs = list(tts.infer(text_,
                              stream=stream,
                              params_infer_code=params_infer_code,
                              do_text_normalization=True,
                              use_decoder=True,
                              ))

        # zip all the audio files together
        buf = io.BytesIO()

        add_wavs_to_zip(buf, wavs, stream, text)

        # Reset the buf to the start
        buf.seek(0)

        return StreamingResponse(buf, media_type="application/zip")

    # Tag: V2T
    @app.post("/v2t/transform",
              tags=["V2T"],
              summary="voice to text转换",
              )
    async def v2t(audio_file: UploadFile = File(..., description="用户输入语音文件")):
        # 将上传的文件保存到临时文件
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(audio_file.file.read())
            tmp_path = tmp.name

        audio, sampling_rate = librosa.load(tmp_path, sr=16_000)

        input_features = v2t_processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to("cuda")
        prompt_ids = v2t_processor.get_prompt_ids("以下是普通话的句子:", return_tensors="pt").to("cuda")

        predicted_ids = v2t_model.generate(input_features=input_features, prompt_ids=prompt_ids)
        transcription = v2t_processor.batch_decode(predicted_ids, skip_special_tokens=True)

        return transcription[0][len(prompt_ids) - 1:]

    # Tag: T2V
    @app.post("/t2v/transform",
              tags=["T2V"],
              summary="text to voice转换",
              )
    async def t2v(text: str = Body(..., description="文字输入", examples=["你好，早上好！"]),
                  stream: Optional[bool] = Body(False, description="流式输出"),
                  speed: Optional[str] = Body("[speed_2]", description="语速"),
                  seed: Optional[int] = Body(300, description="音色")
                  ):
        # std, mean = chat.sample_random_speaker
        torch.manual_seed(seed)
        rand_spk = tts.sample_random_speaker()

        params_infer_code = ChatTTS.Chat.InferCodeParams(
            prompt=speed,
            spk_emb=rand_spk,  # add sampled speaker
            temperature=.3,  # using custom temperature
        )

        params_refine_text = ChatTTS.Chat.RefineTextParams(
            prompt='[oral_2][laugh_0][break_6]',
        )

        wavs = list(tts.infer(text,
                              stream=stream,
                              params_infer_code=params_infer_code,
                              do_text_normalization=True,
                              use_decoder=True,
                              ))

        # zip all the audio files together
        buf = io.BytesIO()

        add_wavs_to_zip(buf, wavs, stream)

        # Reset the buf to the start
        buf.seek(0)

        return StreamingResponse(buf, media_type="application/zip")


def run_api(host, port, **kwargs):
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='',
                                     description='')
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)

    # 初始化消息
    args = parser.parse_args()
    args_dict = vars(args)

    app = create_app()

    run_api(host=args.host,
            port=args.port,
            )
