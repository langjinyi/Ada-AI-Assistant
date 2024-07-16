# _*_ coding: utf-8 _*_
# @Time    : 2024/7/15 21:00
# @Author  : JINYI LIANG
import requests

# Define the endpoint URL
url = 'http://localhost:7861/voice_text/chat'

# Prepare the data to be sent in the request
data = {
    "text_query": "你好",
    "history": [
    str({"role": "user", "content": "Hello, how are you?"}),
    str({"role": "assistant", "content": "I'm fine, thank you."})
],
    "K": 3,
    "stream": False,
    "temperature": 1.0,
    "max_tokens": 100,
    "prompt_name": "default",
    "speed": "[speed_2]",
    "seed": 300
}

# Make the POST request
response = requests.post(url, data=data)

# Print the response from the server
print(response)
