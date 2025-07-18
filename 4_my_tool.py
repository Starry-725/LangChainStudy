'''
Descripttion: 说明
version: V1.0
Author: StarryLei
Date: 2025-07-17 22:46:27
LastEditors: StarryLei
LastEditTime: 2025-07-18 01:32:36
'''
import os
import requests
import json
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv(override=True)

ARK_API_KEY = os.getenv("ARK_API_KEY")
# 确保 API 密钥已设置
if "ARK_API_KEY" not in os.environ:
    raise ValueError("请设置环境变量 ARK_API_KEY")

def print_chain_out(x):
    print("中间输出的结果为：",x)
    return x

chatModel = ChatOpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=ARK_API_KEY,
        model="deepseek-r1-250120",
        streaming=True
    )

# 写一个自己的调用自定义天气查询工具的链

debug_node = RunnableLambda(print_chain_out)

@tool
def get_weather(loc):
    """
    查询即时天气函数
    :param loc: 必要参数，字符串类型，用于表示查询天气的具体城市名称，\
    注意，中国的城市需要用对应城市的英文名称代替，例如如果需要查询北京市天气，则loc参数需要输入'Beijing'；
    :return：OpenWeather API查询即时天气的结果，具体URL请求地址为：https://api.openweathermap.org/data/2.5/weather\
    返回结果对象类型为解析之后的JSON格式对象，并用字符串形式进行表示，其中包含了全部重要的天气信息
    """
    # Step 1.构建请求
    url = "https://api.openweathermap.org/data/2.5/weather"

    # Step 2.设置查询参数
    params = {
        "q": loc,               
        "appid": os.getenv("OPENWEATHER_API_KEY"),    # 输入API key
        "units": "metric",            # 使用摄氏度而不是华氏度
        "lang":"zh_cn"                # 输出语言为简体中文
    }

    # Step 3.发送GET请求
    response = requests.get(url, params=params)
    
    # Step 4.解析响应
    data = response.json()
    return json.dumps(data)

# 将自定义的python函数通过bind_tools绑定到大模型中
tools = [get_weather]
llm_with_tools = chatModel.bind_tools(tools)

# first_tool_only: 这个参数是一个布尔值，用于处理模型可能对同一个工具要求调用多次的情况。
# 如果模型因为某种原因，返回了多个对 get_weather 工具的调用请求（例如 get_weather(location='北京') 和 get_weather(location='上海')），这个解析器也只会提取并返回第一个。
parser = JsonOutputKeyToolsParser(key_name=get_weather.name,first_tool_only=True)

# 链路顺序：先套用模板的提示词，接入绑定查询天气工具的大模型，对模型输出的json进行解析本链关注的工具，对大模型返回结果执行get_weather方法得到结果
query_weather_chain = llm_with_tools | parser | get_weather
# print(query_weather_chain.invoke("请问今天上海的天气怎么样？"))

# 构建问答链
response_prompt = PromptTemplate.from_template(
    """你将收到一段 JSON 格式的天气数据，请用简洁自然的方式将其转述给用户。
以下是天气 JSON 数据：

```json
{weather_json}
```

请将其转换为中文天气描述，例如：
“北京当前天气晴，气温为 23°C，湿度 58%，风速 2.1 米/秒。”
只返回一句话描述，不要其他说明或解释。"""
)

response_chain = response_prompt | chatModel | StrOutputParser()

overall_chain = query_weather_chain | response_chain

print(overall_chain.invoke("请问今天南京的天气怎么样？"))

