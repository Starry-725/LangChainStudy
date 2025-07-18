'''
Author: Starry 1018485883@qq.com
Date: 2025-07-18 16:41:22
LastEditors: Starry 1018485883@qq.com
LastEditTime: 2025-07-18 18:19:40
FilePath: /LangChainStudy/5_my_agent.py
Description: 
'''
import os
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor,create_tool_calling_agent,tool

load_dotenv(override=True)

search = TavilySearchResults(max_result=5)
print(search.invoke("盘古大模型是造假么"))