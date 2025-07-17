import os
import uuid
import gradio as gr

# --- 1. 修复 LangChain 导入 ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory # <--- 已修改
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import SystemMessage

# 确保 API 密钥已设置
if "ARK_API_KEY" not in os.environ:
    raise ValueError("请设置环境变量 ARK_API_KEY")

# 初始化模型
chatARK = ChatOpenAI(
    model="deepseek-r1-250120",
    api_key=os.getenv("ARK_API_KEY"),
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    temperature=0.7,
    streaming=True,
)

# 创建提示模板
# 注意：为了适配 Gradio 的 'messages' 格式，我们稍微调整一下，确保 system prompt 能被正确处理
# 实际上 RunnableWithMessageHistory 会自动处理，但这样写更清晰
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# 创建输出解析器
output_parser = StrOutputParser()

# 构建核心链
# 我们在链的最开始加入 SystemMessage，这样它总是在对话的最前面
chain_with_sys_prompt = (
    SystemMessage(content="你叫Starry，是一名乐于助人的人工智能助手，请用中文回答所有问题。")
    + prompt
)
core_chain = chain_with_sys_prompt | chatARK | output_parser

# 设置记忆存储
demo_ephemeral_chat_history_for_chain = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in demo_ephemeral_chat_history_for_chain:
        demo_ephemeral_chat_history_for_chain[session_id] = ChatMessageHistory()
    return demo_ephemeral_chat_history_for_chain[session_id]

# 将核心链包装成带记忆的链
conversational_chain = RunnableWithMessageHistory(
    core_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# --- Gradio 界面部分 ---

def predict(message, session_id: str):
    """Gradio 的核心预测函数"""

    # 流式调用链
    stream = conversational_chain.stream(
        {"input": message},
        config={"configurable": {"session_id": session_id}}
    )

    # 流式返回
    is_first_chunk = True
    for chunk in stream:
        if is_first_chunk and not chunk.strip():
            continue
        if is_first_chunk:
            yield chunk.lstrip()
            is_first_chunk = False
        else:
            yield chunk

# --- 2. Gradio 组件初始化和布局 ---
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="sky"),
    css="#chatbot { min-height: 600px; }"
) as demo:
    session_id_state = gr.State(str(uuid.uuid4()))

    gr.Markdown(
        """
        # 🤖 Starry - 你的AI助手
        由 LangChain 和火山引擎大模型驱动。
        """
    )

    chatbot = gr.Chatbot(
        elem_id="chatbot",
        label="Starry",
        type="messages",
    )

    txt = gr.Textbox(
        show_label=False,
        placeholder="你好，有什么可以帮你的吗？",
        container=False,
    )

    def add_text(history, text):
        # 如果用户没输入内容，则不做任何事
        if not text:
            return history, gr.update(value="", interactive=True)
        history.append({"role": "user", "content": text})
        return history, gr.update(value="", interactive=False)

    def stream_message(history, session_id):
        user_message = history[-1]["content"]
        history[-1]["role"] = "user"
        history.append({"role": "assistant", "content": ""})

        stream = predict(user_message, history, session_id)
        for chunk in stream:
            history[-1]["content"] += chunk
            yield history

    # --- 修复后的事件链 ---
    txt.submit(
        add_text,
        [chatbot, txt],
        [chatbot, txt]
    ).then(
        stream_message,
        [chatbot, session_id_state],
        chatbot
    ).then(
        lambda: gr.update(interactive=True),
        None,
        [txt]
    )

    demo.load(lambda: gr.update(interactive=True), None, [txt])

if __name__ == "__main__":
    demo.launch()
    
    # # 第一种带入历史对话的方法——————————————————————————————————————————————————————————
    # # 创建一个messages_list去存储历史对话
    # messages_list=[]
    # print("🔹 输入 exit 结束对话")
    # while True:
    #     user_query = input("👤 你：")
    #     if user_query.lower() in {"exit", "quit"}:
    #         break

    #     # 1) 追加用户消息
    #     messages_list.append(HumanMessage(content=user_query))

    #     # 2) 调用模型
    #     assistant_reply = core_chain.invoke({"input":user_query,"history": messages_list})
    #     print("🤖 小智：", assistant_reply)

    #     # 3) 追加 AI 回复
    #     messages_list.append(AIMessage(content=assistant_reply))

    #     # 4) 仅保留最近 50 条
    #     messages_list = messages_list[-50:]