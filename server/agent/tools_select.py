# _*_ coding: utf-8 _*_
# @Time    : 2024/7/15 15:07
# @Author  : JINYI LIANG
from langchain.tools import Tool
from server.agent.tools import *
from server.agent.tools.search_internet import search_internet, SearchInternetInput

tools = [
    Tool.from_function(
        func=calculate,
        name="calculate",
        description="Useful for when you need to answer questions about simple calculations",
        args_schema=CalculatorInput,
    ),
    Tool.from_function(
        func=arxiv,
        name="arxiv",
        description="A wrapper around Arxiv.org for searching and retrieving scientific articles in various fields.",
        args_schema=ArxivInput,
    ),
    Tool.from_function(
        func=weathercheck,
        name="weather_check",
        description="查看天气情况",
        args_schema=WeatherInput,
    ),
    Tool.from_function(
        func=search_internet,
        name="search_internet",
        description="Use this tool to use bing search engine to search the internet",
        args_schema=SearchInternetInput,
    ),
]

tool_names = [tool.name for tool in tools]