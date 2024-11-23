# _*_ coding: utf-8 _*_
# @Time    : 2024/7/15 15:07
# @Author  : JINYI LIANG
from langchain.tools import Tool
from server.agent.tools import *

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
        description="Use this tool to check the real-time weather",
        args_schema=WeatherInput,
    ),

    Tool.from_function(
        func=google_search,
        name="internet_search",
        description="Use this tool to use search engine to search the internet",
        args_schema=InternetSearchInput,
    ),

]

tool_names = [tool.name for tool in tools]