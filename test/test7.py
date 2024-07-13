# _*_ coding: utf-8 _*_
# @Time    : 2024/7/10 11:53
# @Author  : JINYI LIANG
import datetime
import json
import os
import zipfile

import requests
import io
from scipy.io import wavfile
import sounddevice as sd

# API的URL
url = 'http://localhost:7861/t2v/transform'  # 假设你的服务器运行在本地的8000端口

# 要发送的文本
text = "今天天气真好我们一起出去玩吧"

headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
}


# Function to play audio from in-memory bytes
def play_audio(audio_bytes):
    # Create a BytesIO buffer from the audio bytes
    audio_buffer = io.BytesIO(audio_bytes)

    # Read the WAV file from the buffer
    sample_rate, audio_data = wavfile.read(audio_buffer)

    # Play the audio
    sd.play(audio_data, samplerate=sample_rate)
    sd.wait()  # Wait until the audio finishes playing


data = {
    "text": text,
    "stream": True
    # "speed": "[speed_9]",
    # "seed": 300
}

# 发送POST请求
response = requests.post(url, json=data, headers=headers)

# Check response status
if response.status_code == 200:
    # Open the ZIP file from the response content
    with zipfile.ZipFile(io.BytesIO(response.content), "r") as zip_ref:
        # Iterate through each file in the ZIP archive
        for file_info in zip_ref.infolist():
            if file_info.filename.endswith('.wav'):
                # Read the file data into memory
                with zip_ref.open(file_info.filename) as file:
                    audio_bytes = file.read()
                    # Play the audio data
                    play_audio(audio_bytes)
else:
    print("Failed to call API, status code:", response.status_code)
