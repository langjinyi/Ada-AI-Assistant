# _*_ coding: utf-8 _*_
# @Time    : 2024/7/5 14:31
# @Author  : JINYI LIANG

import os

# 可以指定一个绝对路径，统一存放所有的Embedding和LLM模型。
# 每个模型可以是一个单独的目录，也可以是某个目录下的二级子目录。
# 如果模型目录名称和 MODEL_PATH 中的 key 或 value 相同，程序会自动检测加载，无需修改 MODEL_PATH 中的路径。
MODEL_ROOT_PATH = ""

# 要运行的 LLM 名称，可以包括本地模型和在线模型。列表中本地模型将在启动项目时全部加载。

LLM_MODELS = "Qwen-7B-Chat"
V2T_MODELS = "whisper-tiny"
T2V_MODELS = "chatTTS"

# LLM 模型运行设备。设为"auto"会自动检测(会有警告)，也可手动设定为 "cuda","mps","cpu","xpu" 其中之一。
LLM_DEVICE = "auto"

HISTORY_LEN = 3

MAX_TOKENS = 2048

TEMPERATURE = 0.7

MODEL_PATH = {
    "llm_model": {
        "Qwen-7B-Chat": "models/Qwen-7B-Chat/Qwen/Qwen-7B-Chat",
    },
    "v2t_model": {
        "whisper-tiny": "models/whisper-tiny",
    },
    "t2v_model": {
        "chatTTS": "models/chatTTS/pzc163/chatTTS",
    },
}

# nltk 模型存储路径
NLTK_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nltk_data")