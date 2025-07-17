import os
from dotenv import load_dotenv 
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import BooleanOutputParser,ResponseSchema
from langchain.prompts import ChatPromptTemplate

load_dotenv(override=True)

ARK_API_KEY = os.getenv("ARK_API_KEY")
dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")


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
print("-----调用火山引擎的deepseek_r1进行回答-----")
# StrOutputParser会自动解析接口返回的content内容
basic_qa_chain = chat | StrOutputParser()
print(basic_qa_chain.invoke("你好，你是谁"))

# 用提示词模板去构建一个延长chain
prompt_template = ChatPromptTemplate([
    ("system","你是一个中文语法，请你判断用户提供的句子是否有语病。"),
    ("user","这是用户需要判断的句子：{sentence}，请用 yes 或者 no 来回答")
])
# 现在我来延长chain
sentence = "中试基地自去年立项以来，就坚持见设与着商‘同步走’的战略。师晓倩介绍，园区举办各种中试基地推介会、项目对接会、路演会等，储备中试项目60余个，“确保中试基地建成即投运，开园即满园，全力打造化工园中试基地标杆。”根据规划方案，中试基地占地156亩，建设16栋甲类中试厂房、1887平方米甲类和丙类仓库，配套撬装中试区、智慧化管理平台、污水处理站、废气治理共享“绿岛”等功能区域。"
prompt_qa_chain = prompt_template | chat | BooleanOutputParser()
result = prompt_qa_chain.invoke(sentence)
print(result)


# DashScope的qwen模型流式调用方法
# 最佳实践：从环境变量读取 API Key
if not dashscope_api_key:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

try:
    # 1. 初始化模型，直接传入 API Key
    chatLLM = ChatTongyi(
        model="qwen-max",   
        dashscope_api_key=dashscope_api_key
    )
    
    # 非流式调用
    # print(chatLLM.invoke("你好").content)
    
    # 2. 使用 .stream() 进行流式调用
    print("-----调用DashScope的Qwen进行流式回答-----")
    chunks = chatLLM.stream([HumanMessage(content="你好,你是谁")])

    # 3. 遍历并打印返回的内容块
    for chunk in chunks:
        # chunk.content 是每个小块的文本内容
        print(chunk.content, end="", flush=True)

    print("\n流式调用结束。")
    

except Exception as e:
    print(f"\n调用时发生错误: {e}")