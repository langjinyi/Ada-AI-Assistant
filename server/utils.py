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
from configs import (LLM_MODELS, LLM_DEVICE, logger, log_verbose,
                     MODEL_PATH, MODEL_ROOT_PATH, ONLINE_LLM_MODEL)
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
    from configs.model_config import MODEL_PATH, ONLINE_LLM_MODEL
    from configs.server_config import FSCHAT_MODEL_WORKERS
    from server import model_workers

    config = FSCHAT_MODEL_WORKERS.get("default", {}).copy()
    config.update(ONLINE_LLM_MODEL.get(model_name, {}).copy())
    config.update(FSCHAT_MODEL_WORKERS.get(model_name, {}).copy())

    # Api大模型
    if model_name in ONLINE_LLM_MODEL:
        config["online_api"] = True
        config["device"] = llm_device(config.get("device"))
        if provider := config.get("provider"):
            try:
                config["worker_class"] = getattr(model_workers, provider)

            except Exception as e:
                msg = f"在线模型 ‘{model_name}’ 的provider没有正确配置"
                logger.error(f'{e.__class__.__name__}: {msg}',
                             exc_info=e if log_verbose else None)

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


def convert_to_int16(audio_data):
    """ Convert float32 audio data to int16. """
    return (audio_data * 32767).astype(np.int16)


def write_wav_to_buffer(audio_data, rate):
    """ Write WAV data to a BytesIO buffer. """
    buffer = io.BytesIO()
    wavwrite(buffer, rate, audio_data)
    buffer.seek(0)
    return buffer



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

    return ''.join(num_to_chinese[digit] for digit in str(number))


def convert_numbers_to_chinese(sentence):
    def number_to_chinese_wrapper(match):
        number = match.group(0)
        return number_to_chinese(number)

    return re.sub(r'\d+', number_to_chinese_wrapper, sentence)
