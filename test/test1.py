# _*_ coding: utf-8 _*_
# @Time    : 2024/7/15 21:00
# @Author  : JINYI LIANG
import io
import json
import os

import requests
from pydub import AudioSegment
from io import BytesIO

from pydub.playback import play

# Define the API endpoint and payload
url = 'http://localhost:7861/chat/chat'
text_data = {
    "text_query": "你好",
}

# Send POST request
response = requests.post(url, data=text_data, stream=True)
# Check if the request was successful
if response.status_code == 200:
    # 打印原始响应内容
    text_data = []
    for chunk in response.iter_content(None, decode_unicode=True):
        data = json.loads(chunk)
        text = data['text']
        audio_data = data['audio'].encode('latin1')  # Encode audio data back to bytes

        # Process the text
        print(f"Text: {text}")

        # Convert audio data from bytes to a suitable format
        audio_bytes = bytes(audio_data)

        # Example: Save audio data to a file (you can also process it differently)
        with open(f"{text}.wav", "wb") as audio_file:
            audio_file.write(audio_bytes)

else:
    print(f"Request failed with status code: {response.status_code}")
