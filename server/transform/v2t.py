# _*_ coding: utf-8 _*_
# @Time    : 2024/7/16 12:25
# @Author  : JINYI LIANG

import tempfile
import librosa
from fastapi import APIRouter, UploadFile, File
from server.models import models

router = APIRouter()


@router.post("/v2t",
             tags=["V2T"],
             summary="voice to text转换")
async def v2t(audio_file: UploadFile = File(..., description="用户输入语音文件")):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(audio_file.file.read())
        tmp_path = tmp.name

    audio, sampling_rate = librosa.load(tmp_path, sr=16_000)

    input_features = models.v2t_processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to("cuda")
    prompt_ids = models.v2t_processor.get_prompt_ids("以下是普通话的句子:", return_tensors="pt").to("cuda")

    predicted_ids = models.v2t_model.generate(input_features=input_features, prompt_ids=prompt_ids)
    transcription = models.v2t_processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription[0][len(prompt_ids) - 1:]
