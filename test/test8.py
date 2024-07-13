# _*_ coding: utf-8 _*_
# @Time    : 2024/7/10 17:18
# @Author  : JINYI LIANG
import io
import zipfile

import requests
import sounddevice as sd
from pydub import AudioSegment
from pydub.playback import play
from scipy.io import wavfile
from scipy.io.wavfile import write

# API的URL
url = 'http://localhost:7861/voice/chat'  # 假设你的服务器运行在本地的8000端口

headers = {
    'accept': 'application/json',
}


def record_audio(duration=5):
    fs = 16000  # 采样率
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    sd.wait()  # 等待录音结束

    # Save recording to a BytesIO object
    audio_buffer = io.BytesIO()
    write(audio_buffer, fs, myrecording)
    audio_buffer.seek(0)  # Rewind the buffer
    return audio_buffer


def play_audio(audio_bytes):
    # Create a BytesIO buffer from the audio bytes
    audio_buffer = io.BytesIO(audio_bytes)

    # Read the WAV file from the buffer
    sample_rate, audio_data = wavfile.read(audio_buffer)

    # Play the audio
    sd.play(audio_data, samplerate=sample_rate)
    sd.wait()  # Wait until the audio finishes playing


while True:
    print("请开始说话....")
    audio_buffer = record_audio(3)  # 录音3秒
    print("结束录音....")
    # 读取WAV文件内容
    files = {'audio_file': ('output.wav', audio_buffer, 'audio/wav')}
    response = requests.post(url, headers=headers, files=files)

    # 检查响应状态码
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
        break
