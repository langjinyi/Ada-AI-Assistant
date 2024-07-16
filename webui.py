import io
import json
import zipfile

import requests
import gradio as gr
import soundfile as sf

# API的URL
url = 'http://localhost:7861/chat/chat'

headers = {
    'accept': 'application/json',
}


def send_text_request(text_input, chat_history, K, temperature, max_tokens):
    audio_data = {
        "text_query": text_input,
        "history": [str(history) for history in chat_history],
        "K": K,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:

        response = requests.post(url, headers=headers, data=audio_data)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content), "r") as zip_ref:
            audio_output, text_output = None, None
            for file_info in zip_ref.infolist():
                if file_info.filename.endswith('.json'):
                    with zip_ref.open(file_info.filename) as file:
                        audio_text_output = json.load(file)
                        text_output = audio_text_output["text"]
                if file_info.filename.endswith('.mp3'):
                    with zip_ref.open(file_info.filename) as file:
                        audio_output = file.read()
            return audio_output, text_output, audio_text_output["input"]
    except (requests.RequestException, zipfile.BadZipFile, KeyError) as e:
        print(f"Audio request failed: {e}")
        return None, None, None


def send_audio_request(audio_input, chat_history, K, temperature, max_tokens, speed, seed):
    sample_rate, audio = audio_input

    audio_data = {
        "history": [str(history) for history in chat_history],
        "K": K,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "speed": speed,
        "seed": seed,
    }

    try:
        bytes_io = io.BytesIO()
        sf.write(bytes_io, audio, sample_rate, format='mp3')
        bytes_io.seek(0)

        files = {'audio_file': ('output.mp3', bytes_io, 'audio/mp3')}
        response = requests.post(url, headers=headers, files=files, data=audio_data)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content), "r") as zip_ref:
            audio_output, text_output = None, None
            for file_info in zip_ref.infolist():
                if file_info.filename.endswith('.json'):
                    with zip_ref.open(file_info.filename) as file:
                        audio_text_output = json.load(file)
                        text_output = audio_text_output["text"]
                if file_info.filename.endswith('.mp3'):
                    with zip_ref.open(file_info.filename) as file:
                        audio_output = file.read()
            return audio_output, text_output, audio_text_output["input"]
    except (requests.RequestException, zipfile.BadZipFile, KeyError) as e:
        print(f"Audio request failed: {e}")
        return None, None, None


def mock_chat_function(text_input, audio_input, chat_history, K, temperature, max_tokens, speed, seed):
    audio_output = None
    if text_input:
        audio_output, text_output, _ = send_text_request(text_input, chat_history, K, temperature, max_tokens)
        if text_output:
            chat_history.extend([
                {"role": "user", "content": text_input},
                {"role": "assistant", "content": text_output}
            ])

    if audio_input:
        audio_output, audio_text_output, input_text = send_audio_request(
            audio_input, chat_history, K, temperature, max_tokens, speed, seed)
        if audio_output and audio_text_output:
            chat_history.extend([
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": audio_text_output}
            ])

    chat_history_display = [(entry["role"], entry["content"]) for entry in chat_history]
    return audio_output, chat_history_display, gr.update(value=''), gr.update(value=None)


def clear_history(chat_history):
    chat_history.clear()
    return [], gr.update(value=''), gr.update(value=None)


gr.Markdown("# Ada WebUI")

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1) as sidebar_left:
            K = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="保留K轮历史")
            temperature = gr.Slider(minimum=0.0, maximum=2.0, value=0.7, step=0.1, label="LLM 采样温度")
            max_tokens = gr.Number(value=50, label="限制LLM生成Token数量")
            speed = gr.Textbox(value="[speed_2]", label="语速0-9")
            seed = gr.Number(value=300, label="音色")

        with gr.Column(scale=4) as main:
            with gr.Row():
                chatbot = gr.Chatbot(label="Chat History")

            with gr.Row():
                audio_output = gr.Audio(
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
                                inputs=[text_input, audio_input, chat_history, K, temperature, max_tokens, speed, seed],
                                outputs=[audio_output, chatbot, text_input, audio_input])

            clear_button.click(fn=clear_history, inputs=[chat_history], outputs=[chatbot, text_input, audio_input])

demo.launch(share=True)