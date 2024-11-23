import argparse
import os

import numpy as np

os.environ["GRADIO_TEMP_DIR"] = "/tmp"
import io
import json

import queue
import time
import wave
import zipfile
from threading import Thread

import pyaudio
import requests
import gradio as gr
import soundfile as sf
from pydub import AudioSegment
from pydub.playback import play
from pydub.utils import which

# API's URL
url_options = ['http://localhost:7861/chat/chat',
               'http://localhost:7861/chat/agent_chat']

tts_options = ["tts_infer", "edge_tts_infer"]

headers = {
    'accept': 'application/json',
}

# AudioSegment.converter = which(r"D:\Anaconda\envs\yolo_env\Lib\site-packages\ffmpeg\bin\ffmpeg.exe")

def audio_is_mp3(audio_data):
    # Simple check, you might need to refine this
    return audio_data.startswith(b'\xff\xf3')
def send_text_request(text_input, chat_history, url, stream, K, temperature, max_tokens, speed, seed, tts_type):
    text_data = {
        "text_query": text_input,
        "history": [str(history) for history in chat_history],
        "stream": stream,
        "K": K,
        "temperature": temperature,
        "speed": speed,
        "seed": int(seed),
        "max_tokens": int(max_tokens),
        "tts_type": tts_type,
    }
    print(text_data)
    try:
        with requests.post(url, headers=headers, data=text_data, stream=True) as response:
            response.raise_for_status()
            text = ''
            audio = None
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    data = json.loads(chunk)
                    text += data.get('text', '')
                    audio = data.get('audio', '').encode('latin1') if data.get('audio') else audio
                    if audio is not None and text is not None:
                        audio_format = 'mp3' if audio_is_mp3(audio) else 'wav'
                        with open(f'tmp/temp_audio.{audio_format}', 'wb') as f:
                            f.write(audio)
                        yield text, f'tmp/temp_audio.{audio_format}'


    except (requests.RequestException, zipfile.BadZipFile, KeyError) as e:
        print(f"Text request failed: {e}")
        yield None, None


def send_audio_request(audio_input, chat_history, url, stream, K, temperature, max_tokens, speed, seed, tts_type):
    sample_rate, audio = audio_input

    audio_data = {
        "history": [str(history) for history in chat_history],
        "stream": stream,
        "K": K,
        "temperature": temperature,
        "max_tokens": int(max_tokens),
        "speed": speed,
        "seed": int(seed),
        "tts_type": tts_type,
    }

    try:
        bytes_io = io.BytesIO()
        sf.write(bytes_io, audio, sample_rate, format='mp3')
        bytes_io.seek(0)

        files = {'audio_file': ('output.mp3', bytes_io, 'audio/mp3')}

        # # 将 BytesIO 数据保存到本地文件
        # with open('output.mp3', 'wb') as f:
        #     f.write(bytes_io.read())

        with requests.post(url, headers=headers, files=files, data=audio_data, stream=True) as response:
            response.raise_for_status()

            text = ''
            audio = None
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    data = json.loads(chunk)
                    query = data.get('input', '')
                    text += data.get('text', '')
                    audio = data.get('audio', '').encode('latin1') if data.get('audio') else audio
                    if audio is not None and text and query:
                        audio_format = 'mp3' if audio_is_mp3(audio) else 'wav'
                        with open(f'tmp/temp_audio.{audio_format}', 'wb') as f:
                            f.write(audio)
                        yield query, text, f'tmp/temp_audio.{audio_format}'


    except (requests.RequestException, zipfile.BadZipFile, KeyError) as e:
        print(f"Audio request failed: {e}")
        yield None, None, None


def mock_chat_function(text_input, audio_input, chat_history, url, stream, K, temperature, max_tokens, speed, seed, tts_type):
    global text_stream, query
    if text_input:
        for text_output, audio_output in send_text_request(text_input, chat_history, url, stream, K, temperature,
                                                           max_tokens, speed, seed, tts_type):
            text_stream = text_output
            audio_stream = audio_output

            chat_history_display = [(entry["role"], entry["content"]) for entry in chat_history]
            chat_history_display.append(("我", text_input))
            chat_history_display.append(("菲菲", text_stream))



            yield audio_stream, chat_history_display, gr.update(value=''), gr.update(value=None)

        if text_stream:
            chat_history.extend([
                {"role": "user", "content": text_input},
                {"role": "assistant", "content": text_stream}
            ])

    if audio_input:
        for query, text_output, audio_output in send_audio_request(
                audio_input, chat_history, url, stream, K, temperature, max_tokens, speed, seed, tts_type):
            text_stream = text_output
            audio_stream = audio_output

            chat_history_display = [(entry["role"], entry["content"]) for entry in chat_history]
            chat_history_display.append(("user", query))
            chat_history_display.append(("assistant", text_stream))

            yield audio_stream, chat_history_display, gr.update(value=''), gr.update(value=None)


        if text_stream and query:
            chat_history.extend([
                {"role": "user", "content": query},
                {"role": "assistant", "content": text_stream}
            ])


def clear_history(chat_history):
    chat_history.clear()
    return gr.update(value=None), [], gr.update(value=''), gr.update(value=None)


gr.Markdown("# Ada WebUI")

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1) as sidebar_left:
            url = gr.Dropdown(choices=url_options, label="选择 URL", value=url_options[0])
            tts_type = gr.Dropdown(choices=tts_options, label="TTS引擎类型", value=tts_options[0])
            stream = gr.Checkbox(False, label="是否流式输出")
            K = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="保留K轮历史")
            temperature = gr.Slider(minimum=0.0, maximum=2.0, value=0.5, step=0.1, label="LLM 采样温度")
            max_tokens = gr.Number(value=50, label="限制LLM生成Token数量")
            speed = gr.Textbox(value="[speed_2]", label="语速0-9")
            seed = gr.Number(value=1506, label="音色")

        with gr.Column(scale=4) as main:
            with gr.Row():
                chatbot = gr.Chatbot(label="Chat History")

            with gr.Row():
                audio_out = gr.Audio(
                    label="Output Audio",
                    value=None,
                    streaming=True,
                    autoplay=True,
                    interactive=False,
                    )

            with gr.Row():
                text_input = gr.Textbox(
                    label="Input Text",
                    lines=4,
                    placeholder="Please Input Text...",
                )
                audio_input = gr.Audio(source="microphone", label="语音输入", format='mp3')
                # audio_input = gr.Audio(source="microphone", label="语音输入", format='mp3')
            submit_button = gr.Button("发送")
            clear_button = gr.Button("清空历史记录")

            chat_history = gr.State([])

            submit_button.click(fn=mock_chat_function,
                                inputs=[text_input, audio_input, chat_history, url, stream, K, temperature, max_tokens,
                                        speed, seed, tts_type],
                                outputs=[audio_out, chatbot, text_input, audio_input])

            clear_button.click(fn=clear_history, inputs=[chat_history],
                               outputs=[audio_out, chatbot, text_input, audio_input])
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--server_name', type=str)
    parser.add_argument('--server_port', type=int)
    args = parser.parse_args()

    demo.queue()
    demo.launch(share=True, server_name=args.server_name, server_port=args.server_port)
