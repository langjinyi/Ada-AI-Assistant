# _*_ coding: utf-8 _*_
# @Time    : 2024/7/14 23:24
# @Author  : JINYI LIANG

import asyncio
import aiohttp
import time
import json

# 测试URL
URL = 'http://localhost:7861/voice_text/chat'

# 模拟音频文件数据
AUDIO_FILE_PATH = "output.wav"


# 创建模拟的请求数据
async def create_request_data():
    with open(AUDIO_FILE_PATH, "rb") as f:
        audio_data = f.read()
    form_data = aiohttp.FormData()
    form_data.add_field('audio_file', audio_data, filename='test.wav')
    return form_data


# 发送请求并获取响应
async def send_request(session, form_data):
    async with session.post(URL, data=form_data) as response:
        result = await response.read()
        return response.status, result


# 主函数，用于并发发送多个请求
async def main():
    num_requests = 10  # 并发请求数量
    async with aiohttp.ClientSession() as session:
        tasks = []
        for _ in range(num_requests):
            form_data = await create_request_data()
            tasks.append(send_request(session, form_data))

        start_time = time.time()
        responses = await asyncio.gather(*tasks)
        end_time = time.time()

        for status, result in responses:
            print(f"Status: {status}, Result: {len(result)} bytes")

        print(f"Total time for {num_requests} requests: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())
