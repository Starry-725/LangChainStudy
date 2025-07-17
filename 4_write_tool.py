'''
Descripttion: 说明
version: V1.0
Author: StarryLei
Date: 2025-07-17 22:46:27
LastEditors: StarryLei
LastEditTime: 2025-07-18 01:32:36
'''
import os
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

ARK_API_KEY = os.getenv("ARK_API_KEY")
# 确保 API 密钥已设置
if "ARK_API_KEY" not in os.environ:
    raise ValueError("请设置环境变量 ARK_API_KEY")

chatModel = ChatOpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=ARK_API_KEY,
    model="deepseek-r1-250120",
    streaming=True
)

# 创建提示词模板
