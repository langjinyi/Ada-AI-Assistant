# _*_ coding: utf-8 _*_
# @Time    : 2024/7/16 18:53
# @Author  : JINYI LIANG
import requests


url = "https://api.seniverse.com/v3/weather/now.json?key=SfYdoqNuCW39UQUvb&location=佛山&language=zh-Hans&unit=c"
response = requests.get(url)
if response.status_code == 200:
    data = response.json()
    weather = {
        "temperature": data["results"][0]["now"]["temperature"],
        "description": data["results"][0]["now"]["text"],
    }
    print(weather)
