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

# API's URL
url = 'http://localhost:7861/chat/agent_chat'

headers = {
    'accept': 'application/json',
}


def send_text_request(text_input, chat_history, stream, K, temperature, max_tokens):
    text_data = {
        "text_query": text_input,
        "history": [str(history) for history in chat_history],
        "stream": stream,
        "K": K,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

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

                    yield text, audio

    except (requests.RequestException, zipfile.BadZipFile, KeyError) as e:
        print(f"Text request failed: {e}")
        yield None, None


def send_audio_request(audio_input, chat_history, stream, K, temperature, max_tokens, speed, seed):
    sample_rate, audio = audio_input

    audio_data = {
        "history": [str(history) for history in chat_history],
        "stream": stream,
        "K": K,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "speed": speed,
        "seed": seed,
    }

    try:
        bytes_io = io.BytesIO()
        sf.write(bytes_io, audio, sample_rate, format='wav')
        bytes_io.seek(0)

        files = {'audio_file': ('output.wav', bytes_io, 'audio/wav')}
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

                    yield query, text, audio

    except (requests.RequestException, zipfile.BadZipFile, KeyError) as e:
        print(f"Audio request failed: {e}")
        yield None, None, None


def mock_chat_function(text_input, audio_input, chat_history, stream, K, temperature, max_tokens, speed, seed):
    if text_input:
        text_stream, audio_stream = '', None
        for text_output, audio_output in send_text_request(text_input, chat_history, stream, K, temperature,
                                             max_tokens):

            text_stream = text_output if text_output else text_stream
            audio_stream = audio_output if audio_output else audio_stream

            chat_history_display = [(entry["role"], entry["content"]) for entry in chat_history]
            chat_history_display.append(("我", text_input))
            chat_history_display.append(("菲菲", text_stream))
            # time.sleep(1)
            yield audio_stream, chat_history_display, gr.update(value=''), gr.update(value=None)

        if text_stream:
            chat_history.extend([
                {"role": "user", "content": text_input},
                {"role": "assistant", "content": text_stream}
            ])

    if audio_input:
        query, audio_stream, text_stream = '', None, ''
        for query, text_output, audio_output in send_audio_request(
                audio_input, chat_history, stream, K, temperature, max_tokens, speed, seed):

            text_stream = text_output if text_output else text_stream
            audio_stream = audio_output if audio_output else audio_stream

            chat_history_display = [(entry["role"], entry["content"]) for entry in chat_history]
            chat_history_display.append(("我", query))
            chat_history_display.append(("菲菲", text_stream))
            # time.sleep(1)

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
            stream = gr.Checkbox(False, label="是否流式输出")
            K = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="保留K轮历史")
            temperature = gr.Slider(minimum=0.0, maximum=2.0, value=0.7, step=0.1, label="LLM 采样温度")
            max_tokens = gr.Number(value=1024, label="限制LLM生成Token数量")
            speed = gr.Textbox(value="[speed_2]", label="语速0-9")
            seed = gr.Number(value=1506, label="音色")

        with gr.Column(scale=4) as main:
            with gr.Row():
                chatbot = gr.Chatbot(label="Chat History")

            with gr.Row():
                audio_out = gr.Audio(
                    label="Output Audio",
                    format="mp3",
                    autoplay=True,
                    interactive=False,
                )

            with gr.Row():
                text_input = gr.Textbox(
                    label="Input Text",
                    lines=4,
                    placeholder="Please Input Text...",
                )
                audio_input = gr.Audio(sources=["microphone"], label="语音输入", format='mp3')

            submit_button = gr.Button("发送")
            clear_button = gr.Button("清空历史记录")

            chat_history = gr.State([])

            submit_button.click(fn=mock_chat_function,
                                inputs=[text_input, audio_input, chat_history, stream, K, temperature, max_tokens,
                                        speed, seed],
                                outputs=[audio_out, chatbot, text_input, audio_input])

            clear_button.click(fn=clear_history, inputs=[chat_history], outputs=[audio_out, chatbot, text_input, audio_input])

demo.launch(share=True)
