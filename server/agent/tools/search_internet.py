# _*_ coding: utf-8 _*_
# @Time    : 2024/7/26 18:30
# @Author  : JINYI LIANG
from pydantic import BaseModel, Field
import requests
from configs import GOODLE_SEARCH_API_KEY


def google_search(query):
    url = "https://serpapi.com/search"

    params = {
        "q": query,
        "api_key": GOODLE_SEARCH_API_KEY,
        "hl": "zh-cn",
        "num": 5
    }
    response = requests.get(url, params=params)
    print("**********************requested******************")
    if response.status_code == 200:
        data = response.json().get("organic_results", [])
        information = [{
            "title": item["title"],
            "link": item["link"],
            "snippet": item["snippet"],

        }for item in data]
        return information
    else:
        raise Exception(
            f"Failed to retrieve internet: {response.status_code}")


class InternetSearchInput(BaseModel):
    query: str = Field(description="The search query title")
