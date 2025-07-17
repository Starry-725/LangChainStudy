import os
from dotenv import load_dotenv 
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models.volcengine_maas import VolcEngineMaasChat
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage

load_dotenv(override=True)

# 从环境变量读取 API Key
ARK_API_KEY = os.getenv("ARK_API_KEY")
dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")

# 这是调用一些通用标准化API接口的方法，如openai,deepseek等提供的官方接口
# model = init_chat_model(model="deepseek-chat",model_provider="deepseek")


# 重写一个方法去调用类似于火山引擎、硅基流动等官方接口的方法
# 1. 初始化 ChatOpenAI 模型
# 将其配置为指向火山引擎的服务器
chat = ChatOpenAI(
    # model: 从您的 curl 命令中获取
    model="deepseek-r1-250120", 
    # api_key: 粘贴您在上一步中从火山引擎控制台生成的【API Key】
    api_key=os.getenv("ARK_API_KEY"), # <--- 在这里粘贴你的新密钥
    # base_url: 从您的 curl 命令中获取，去掉末尾的 /chat/completions
    base_url="https://ark.cn-beijing.volces.com/api/v3", 
    temperature=0.7,
    # 如果需要，可以禁用流式输出等
    streaming=False,
)
# 2. 准备消息
messages = [
    SystemMessage(content="你是人工智能助手."),
    HumanMessage(content="一加一等于几")
]
# 发起调用并获取回复
response = chat.invoke(messages)
print(response)


# DashScope的qwen模型流式调用方法
if not dashscope_api_key:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

try:
    # 1. 初始化模型，直接传入 API Key
    chatLLM = ChatTongyi(
        model="qwen-max",   
        dashscope_api_key=dashscope_api_key
    )
    
    # 非流式调用
    print(chatLLM.invoke("你好").content)
    
    # 2. 使用 .stream() 进行流式调用
    print("正在进行流式调用...")
    chunks = chatLLM.stream([HumanMessage(content="你好")])

    # 3. 遍历并打印返回的内容块
    for chunk in chunks:
        # chunk.content 是每个小块的文本内容
        print(chunk.content, end="", flush=True)

    print("\n流式调用结束。")
    

except Exception as e:
    print(f"\n调用时发生错误: {e}")