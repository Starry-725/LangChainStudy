import os
from dotenv import load_dotenv 
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import BooleanOutputParser,ResponseSchema,StructuredOutputParser
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate

load_dotenv(override=True)

ARK_API_KEY = os.getenv("ARK_API_KEY")
dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")


# 将其配置为指向火山引擎的服务器
chatARK = ChatOpenAI(
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

# StrOutputParser会自动解析接口返回的content内容
basic_qa_chain = chatARK | StrOutputParser()
print(basic_qa_chain.invoke("你好，你是谁"))


# 用提示词模板去构建一个延长chain---------------------------------------------
prompt_template = ChatPromptTemplate([
    ("system","你是一个中文语法，请你判断用户提供的句子是否有语病。"),
    ("user","这是用户需要判断的句子：{sentence}，请用 yes 或者 no 来回答")
])
# 现在我来延长chain
sentence = "中试基地自去年立项以来，就坚持见设与着商‘同步走’的战略。"
prompt_qa_chain = prompt_template | chatARK | BooleanOutputParser()
result = prompt_qa_chain.invoke(sentence)
print(result)


# DashScope的qwen模型调用方法----------------------------------------------------------
# 最佳实践：从环境变量读取 API Key
if not dashscope_api_key:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

try:
    # 1. 初始化模型，直接传入 API Key
    chatQwen = ChatTongyi(
        model="qwen-max",   
        dashscope_api_key=dashscope_api_key
    )
    
    # 定义我们想要的输出结构 (Define the output structure)
    # 每个 ResponseSchema 对应最终字典中的一个键值对
    response_schemas = [
        ResponseSchema(
            name="sentiment",
            description="这篇评论的情感是积极(positive), 消极(negative)还是中性(neutral)?"
        ),
        ResponseSchema(
            name="summary",
            description="用一句话简短总结这篇评论的主要观点。"
        ),
        ResponseSchema(
            name="suggested_action",
            description="作为客服，针对这条评论，我们应该采取什么后续行动？例如：'联系用户解决问题' 或 '感谢用户的支持'。"
        ),
    ]
    
    # 创建结构化输出解析器 (Create the output parser)
    # StructuredOutputParser 会将 response_schemas 转换为格式化指令
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    
    # 获取格式化指令 (Get the format instructions)
    # 这是告诉 LLM 如何格式化其输出的模板
    format_instructions = output_parser.get_format_instructions()
    # print("format_instructions",format_instructions)
    
    # 4. 创建提示模板 (Create the prompt template)
    # 注意，模板中包含了 {review} (我们的输入) 和 {format_instructions} (解析器的指令)
    prompt = PromptTemplate(
        template="请分析以下的用户评论。\n{format_instructions}\n评论内容: {review}",
        input_variables=["review"],
        partial_variables={"format_instructions": format_instructions}
    )
    
    customer_review = "这款新出的智能手表太棒了！电池续航能力超出了我的预期，能用整整三天。"
    
    analysis_chain = prompt | chatQwen | output_parser
    # 调用链，传入我们的评论
    result = analysis_chain.invoke({"review": customer_review})
    print(result)
  
except Exception as e:
    print(f"\n调用时发生错误: {e}")
    
    
# 创建复合链对商品评论做自动回复——————————————————————————————————————————————————————————————————————————————-
# --- 第二环：回复生成链 ---
reply_template = """
你是一名专业的客服。请根据以下信息，草拟一条礼貌、专业的回复评论。

评论信息：
评论概括: {summary}
评论情感: {sentiment}
回复建议：{suggested_action}

请在回复中根据评论的评论概括、评论情感、回复建议为客户提供相应的回复，不需要任何注释信息。
"""

reply_prompt = ChatPromptTemplate.from_template(reply_template)
reply_chain = reply_prompt | chatARK | StrOutputParser()

overall_chain = analysis_chain | reply_chain

customer_review = "我上周买的你们的‘星辰Pro’智能手表，用了没几天电池就不行了，一天都撑不住，太让人失望了。"
final_reply = overall_chain.invoke({"review": customer_review})
print(final_reply)

# 可以自定义一个chain去查看结果—————————————————————————————————————————————————————————————————————————————-
from langchain_core.runnables import RunnableLambda

# 自定义一个chain用于打印中间结果
def debug_print(x):
    print("debug_print显示的用户评论提取json:",x)
    return x

debug_chain = RunnableLambda(debug_print)

overall_chain = analysis_chain | debug_chain | reply_chain
final_reply = overall_chain.invoke({"review": customer_review})
print(final_reply)