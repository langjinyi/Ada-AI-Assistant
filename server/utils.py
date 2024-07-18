# _*_ coding: utf-8 _*_
# @Time    : 2024/7/8 13:20
# @Author  : JINYI LIANG
import json
import os
import re
import numpy as np
import pydantic
from pydantic import BaseModel
from fastapi import FastAPI
from configs import (LLM_MODELS, LLM_DEVICE,
                     MODEL_PATH, MODEL_ROOT_PATH)
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Literal,
    Optional,
    Callable,
    Generator,
    Dict,
    Any,
    Awaitable,
    Union,
    Tuple
)
import io
import zipfile
from scipy.io.wavfile import write as wavwrite


def fschat_controller_address() -> str:
    from configs.server_config import FSCHAT_CONTROLLER

    host = FSCHAT_CONTROLLER["host"]
    if host == "0.0.0.0":
        host = "127.0.0.1"
    port = FSCHAT_CONTROLLER["port"]
    return f"http://{host}:{port}"


def fschat_model_worker_address(model_name: str = LLM_MODELS[0]) -> str:
    if model := get_model_worker_config(model_name):
        host = model["host"]
        if host == "0.0.0.0":
            host = "127.0.0.1"
        port = model["port"]
        return f"http://{host}:{port}"
    return ""


def fschat_openai_api_address() -> str:
    from configs.server_config import FSCHAT_OPENAI_API

    host = FSCHAT_OPENAI_API["host"]
    if host == "0.0.0.0":
        host = "127.0.0.1"
    port = FSCHAT_OPENAI_API["port"]
    return f"http://{host}:{port}/v1"


def api_address() -> str:
    from configs.server_config import API_SERVER

    host = API_SERVER["host"]
    if host == "0.0.0.0":
        host = "127.0.0.1"
    port = API_SERVER["port"]
    return f"http://{host}:{port}"


# 从server_config中获取服务信息
def get_model_worker_config(model_name: str = None) -> dict:
    '''
    加载model worker的配置项。
    优先级:FSCHAT_MODEL_WORKERS[model_name] > ONLINE_LLM_MODEL[model_name] > FSCHAT_MODEL_WORKERS["default"]
    '''
    from configs.model_config import MODEL_PATH
    from configs.server_config import FSCHAT_MODEL_WORKERS

    config = FSCHAT_MODEL_WORKERS.get("default", {}).copy()
    config.update(FSCHAT_MODEL_WORKERS.get(model_name, {}).copy())

    # 本地模型
    if model_name in MODEL_PATH["llm_model"]:
        path = get_model_path(model_name)
        config["model_path"] = path
        if path and os.path.isdir(path):
            config["model_path_exists"] = True
        config["device"] = llm_device(config.get("device"))
    return config


def get_model_path(model_name: str, type: str = None) -> Optional[str]:
    if type in MODEL_PATH:
        paths = MODEL_PATH[type]
    else:
        paths = {}
        for v in MODEL_PATH.values():
            paths.update(v)

    if path_str := paths.get(model_name):  # 以 "chatglm-6b": "THUDM/chatglm-6b-new" 为例，以下都是支持的路径
        path = Path(path_str)
        if path.is_dir():  # 任意绝对路径
            return str(path)

        root_path = Path(MODEL_ROOT_PATH)
        if root_path.is_dir():
            path = root_path / model_name
            if path.is_dir():  # use key, {MODEL_ROOT_PATH}/chatglm-6b
                return str(path)
            path = root_path / path_str
            if path.is_dir():  # use value, {MODEL_ROOT_PATH}/THUDM/chatglm-6b-new
                return str(path)
            path = root_path / path_str.split("/")[-1]
            if path.is_dir():  # use value split by "/", {MODEL_ROOT_PATH}/chatglm-6b-new
                return str(path)
        return path_str  # THUDM/chatglm06b


def llm_device(device: str = None) -> Literal["cuda", "mps", "cpu"]:
    device = device or LLM_DEVICE
    if device not in ["cuda", "mps", "cpu"]:
        device = detect_device()
    return device


def detect_device() -> Literal["cuda", "mps", "cpu"]:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except:
        pass
    return "cpu"


class BaseResponse(BaseModel):
    code: int = pydantic.Field(200, description="API status code")
    msg: str = pydantic.Field("success", description="API status message")
    data: Any = pydantic.Field(None, description="API data")

    class Config:
        schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
            }
        }


def remove_special_characters(text):
    # 使用正则表达式替换非字母和非数字字符
    pattern = r'\（[^）]*\）|\([^\)]*\)'
    # 使用正则表达式替换掉所有匹配的圆括号及其内容
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text


def unsafe_float_to_int16(wav: np.ndarray) -> bytes:
    # 将浮点数转换为 [-32768, 32767] 范围内的整数
    int16_wav = np.int16(wav * 32767)
    # 将整数数组转换为字节数组
    return int16_wav.tobytes()


def convert_to_int16(audio_data):
    """ Convert float32 audio data to int16. """
    return (audio_data * 32767).astype(np.int16)


def write_wav_to_buffer(audio_data, rate):
    """ Write WAV data to a BytesIO buffer. """
    buffer = io.BytesIO()
    wavwrite(buffer, rate, audio_data)
    buffer.seek(0)
    return buffer


def add_wavs_to_zip(zip_file, wavs, stream, text=None):
    """ Add WAV files to a zip archive. """
    with zipfile.ZipFile(zip_file, "a", compression=zipfile.ZIP_DEFLATED, allowZip64=False) as zf:
        # Write the JSON file to the zip
        json_buffer = io.BytesIO()
        json_buffer.write(json.dumps(text, ensure_ascii=False, indent=4).encode('utf-8'))
        json_buffer.seek(0)
        zf.writestr("text.json", json_buffer.read())

        if stream:
            for idx, wav in enumerate(wavs):
                audio_data = convert_to_int16(np.array(wav, dtype=np.float32))
                wav_buffer = write_wav_to_buffer(audio_data, rate=24000)
                zf.writestr(f"{idx}.mp3", wav_buffer.read())
        else:
            audio_data = convert_to_int16(np.array(wavs, dtype=np.float32))
            wav_buffer = write_wav_to_buffer(audio_data, rate=24000)
            zf.writestr("output.mp3", wav_buffer.read())


import re


def number_to_chinese(number):
    num_to_chinese = {
        '0': '零',
        '1': '一',
        '2': '二',
        '3': '三',
        '4': '四',
        '5': '五',
        '6': '六',
        '7': '七',
        '8': '八',
        '9': '九'
    }

    units = ['', '十', '百', '千', '万', '十万', '百万', '千万', '亿', '十亿']

    if number == 0:
        return num_to_chinese['0']

    digits = list(str(number))
    length = len(digits)
    chinese_str = ''
    zero_flag = False

    for i in range(length):
        digit = digits[i]
        pos = length - i - 1
        unit = units[pos]

        if digit == '0':
            zero_flag = True
            if pos % 4 == 0:  # 在万、亿等位置保留零
                chinese_str += unit
        else:
            if zero_flag:
                chinese_str += num_to_chinese['0']
                zero_flag = False
            chinese_str += num_to_chinese[digit] + unit

    return chinese_str.rstrip('零')


def convert_numbers_to_chinese(sentence):
    def number_to_chinese_wrapper(match):
        number = int(match.group(0))
        return number_to_chinese(number)

    return re.sub(r'\d+', number_to_chinese_wrapper, sentence)
