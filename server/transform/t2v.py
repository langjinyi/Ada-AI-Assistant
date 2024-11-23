# _*_ coding: utf-8 _*_
# @Time    : 2024/7/16 12:25
# @Author  : JINYI LIANG

import io

import numpy as np
from fastapi import APIRouter, Body
from fastapi.responses import StreamingResponse
from typing import Optional
import torch
import ChatTTS

from server.models import models
from server.utils import convert_to_int16, write_wav_to_buffer

router = APIRouter()

@router.post("/t2v",
             tags=["T2V"],
             summary="text to voice转换")
async def t2v(text: str = Body(..., description="文字输入", examples=["你好，早上好！"]),
              stream: Optional[bool] = Body(False, description="流式输出"),
              speed: Optional[str] = Body("[speed_2]", description="语速"),
              seed: Optional[int] = Body(300, description="音色")):
    torch.manual_seed(seed)
    rand_spk = models.tts.sample_random_speaker()

    params_infer_code = ChatTTS.Chat.InferCodeParams(
        prompt=speed,
        spk_emb=rand_spk,
        temperature=.3,
    )

    wavs = models.tts.infer(text,
                          stream=stream,
                          params_infer_code=params_infer_code,
                          do_text_normalization=True,
                          use_decoder=True)

    # Convert the audio chunk to int16 format
    audio_data = convert_to_int16(np.array(wavs, dtype=np.float32))

    # Write the audio data to a buffer
    buffer = write_wav_to_buffer(audio_data, rate=24000)

    return StreamingResponse(buffer, media_type="application/json")