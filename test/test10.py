import gradio as gr
import requests
from typing import List


def chat(audio_file=None, text_query="", history="", K=3, stream=False, model_name="LLM_MODELS",
         temperature=1.0, max_tokens=None, prompt_name="default", speed="[speed_2]", seed=1506, tts_type="tts_infer"):
    url = "http://localhost:7861/chat/chat"  # 替换为您的API的实际URL
    headers = {'accept': 'application/json'}

    # 准备请求数据
    data = {
        "text_query": text_query,
        "history": [""],

    }

    files = {"audio_file": (audio_file.name, audio_file, "audio/wav")} if audio_file else None

    response = requests.post(url, headers=headers, data=data, files=files)

    try:
        return response.json()
    except ValueError:
        return response.text


# Gradio界面
interface = gr.Interface(
    fn=chat,
    inputs=[
        gr.Audio(source="upload", label="用户输入语音文件"),
        gr.Textbox(lines=2, placeholder="用户输入文字", label="用户输入文字"),
        gr.Textbox(lines=5, placeholder="用户聊天记录 (以换行符分隔)", label="用户聊天记录"),
        gr.Slider(minimum=1, maximum=10, step=1, value=3, label="保留K轮历史"),
        gr.Checkbox(value=False, label="流式输出"),
        gr.Dropdown(choices=["LLM_MODELS"], value="LLM_MODELS", label="LLM 模型名称"),
        gr.Slider(minimum=0.0, maximum=2.0, step=0.1, value=1.0, label="LLM 采样温度"),
        gr.Number(value=None, label="限制LLM生成Token数量"),
        gr.Textbox(value="default", label="使用的prompt模板名称"),
        gr.Dropdown(choices=["[speed_2]", "[speed_1]", "[speed_3]"], value="[speed_2]", label="语速"),
        gr.Number(value=1506, label="音色"),
        gr.Textbox(value="tts_infer", label="TTS引擎类型")
    ],
    outputs="json"
)

interface.launch(server_name="0.0.0.0", server_port=8001)
