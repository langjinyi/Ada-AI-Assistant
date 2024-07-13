# _*_ coding: utf-8 _*_
# @Time    : 2024/7/9 21:41
# @Author  : JINYI LIANG
import io

import librosa
import requests
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
# from server.utils import api_address

from pprint import pprint
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import speech_recognition as sr

import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

api_base_url = "http://localhost:7861"


def record_audio(duration, sample_rate=44100):
    print("请开始说话....")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("结束录音....")

    # Save to BytesIO instead of a file
    audio_buffer = io.BytesIO()
    write(audio_buffer, sample_rate, recording)  # Write the recording to the buffer
    audio_buffer.seek(0)  # Rewind the buffer to the beginning
    return audio_buffer


def dump_input(d, title):
    print("\n")
    print("=" * 30 + title + "  input " + "=" * 30)
    pprint(d)


def dump_output(r, title):
    print("\n")
    print("=" * 30 + title + "  output" + "=" * 30)
    for line in r.iter_content(None, decode_unicode=True):
        print(line, end="", flush=True)


headers = {
    'accept': 'application/json',
}


def v2t_test(api="/v2t/transform"):
    url = f"{api_base_url}{api}"

    audio_buffer = record_audio(5)  # 调用函数，录音5秒

    # Send the buffer to the API
    files = {'audio_file': ('output.wav', audio_buffer, 'audio/wav')}
    response = requests.post(url, headers=headers, files=files)

    return response.text


def test_thread():
    threads = []
    times = []
    pool = ThreadPoolExecutor()
    start = time.time()
    for i in range(10):
        t = pool.submit(v2t_test)
        threads.append(t)

    for r in as_completed(threads):
        end = time.time()
        times.append(end - start)
        print("\nResult:\n")
        pprint(r.result())

    print("\nTime used:\n")
    for x in times:
        print(f"{x}")


result = v2t_test(api="/v2t/transform")
print(result)
