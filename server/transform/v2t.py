# _*_ coding: utf-8 _*_
# @Time    : 2024/7/9 21:13
# @Author  : JINYI LIANG
import tempfile

from fastapi import Body, UploadFile, File
from numpy import ndarray

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
import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration


async def v2t(audio_file: UploadFile = File(..., description="用户输入语音文件")):
    # 将上传的文件保存到临时文件
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(audio_file.file.read())
        tmp_path = tmp.name

    audio, sampling_rate = librosa.load(tmp_path, sr=16_000)
    # load model and processor
    processor = WhisperProcessor.from_pretrained("models/whisper-tiny")
    model = WhisperForConditionalGeneration.from_pretrained("models/whisper-tiny").to("cuda")
    model.config.forced_decoder_ids = None

    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to("cuda")
    prompt_ids = processor.get_prompt_ids("以下是普通话的句子:", return_tensors="pt").to("cuda")

    predicted_ids = model.generate(input_features=input_features, prompt_ids=prompt_ids)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription[0][len(prompt_ids) - 1:]
