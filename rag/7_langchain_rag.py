'''
Author: Starry 1018485883@qq.com
Date: 2025-07-23 17:51:50
LastEditors: Starry 1018485883@qq.com
LastEditTime: 2025-07-24 17:22:06
FilePath: /LangChainStudy/rag/7_langchain_rag.py
Description: 基于langchain和gradio搭建的RAG系统。
'''
import gradio as gr
import os
from dotenv import load_dotenv

# --- LangChain核心模块 ---
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# --- 1. 环境准备 ---

# 加载.env文件中的环境变量
load_dotenv()

# 检查OpenAI API Key是否存在
if not os.getenv("ARK_API_KEY"):
    raise ValueError("请在.env文件中设置ARK_API_KEY")

ARK_API_KEY = os.getenv("ARK_API_KEY")

# --- 2. 核心RAG逻辑封装成一个函数 ---

def create_rag_chain(file_path: str):
    """
    根据上传的PDF文件路径，创建并返回一个完整的RAG链。
    这个函数包含了RAG的所有步骤：加载、分割、嵌入、存储、检索和生成。
    """
    # 步骤1: 加载PDF文档
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # 步骤2: 将文档分割成小块
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 步骤3: 创建向量存储 (使用FAISS)
    # 这会使用OpenAI的嵌入模型将文本块转换为向量，并存储在FAISS索引中
    vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    # 步骤4: 创建LLM和提示模板
    llm = ChatOpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=ARK_API_KEY,
    model="ep-m-20250411184749-5qknb",
    # streaming=True, # 在Agent模式下，流式输出处理更复杂，暂时关闭以简化
    )
    
    # 一个精心设计的提示，指导LLM如何利用上下文回答问题
    prompt = ChatPromptTemplate.from_template("""
    请你根据下面提供的上下文来回答问题。如果你在上下文中找不到答案，就说你不知道。
    请保持回答的简洁和专业。

    <context>
    {context}
    </context>

    Question: {input}
    """)

    # 步骤5: 创建文档处理链和检索链
    # create_stuff_documents_chain: 将检索到的文档“塞入”提示中
    document_chain = create_stuff_documents_chain(llm, prompt)

    # 从向量存储中创建一个检索器，用于获取相关文档
    retriever = vectorstore.as_retriever()

    # create_retrieval_chain: 结合检索器和文档处理链，形成完整的RAG链
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain


# --- 3. Gradio界面逻辑 ---

# 定义一个处理文件上传的函数
# 这个函数只在用户上传新文件时运行一次
def process_file(file):
    if file is None:
        return None, gr.update(value="请先上传一个PDF文件", interactive=False)
    
    print(f"正在处理文件: {file.name}")
    
    # 创建RAG链
    try:
        rag_chain = create_rag_chain(file.name)
        # 返回创建好的RAG链，并更新聊天输入框状态
        return rag_chain, gr.update(value="文件处理完成，可以开始提问了。", interactive=True)
    except Exception as e:
        print(f"处理文件时出错: {e}")
        return None, gr.update(value=f"处理文件失败: {e}", interactive=False)

# 定义一个处理聊天交互的函数
# `history`是Gradio的聊天记录，`rag_chain_state`是我们存储RAG链的状态
def chat_with_doc(message, history, rag_chain_state):
    if rag_chain_state is None:
        return "请先上传并成功处理一个PDF文件。", history
    
    print(f"收到问题: {message}")
    
    # 调用RAG链获取答案
    response = rag_chain_state.invoke({"input": message})
    answer = response["answer"]
    
    print(f"生成的答案: {answer}")
    
    # 将答案添加到聊天记录中
    history.append((message, answer))
    return "", history


# --- 4. 构建Gradio应用 ---

with gr.Blocks(theme=gr.themes.Default(primary_hue="blue")) as demo:
    gr.Markdown("# 欢迎使用文档问答机器人 (Starry RAG)")
    gr.Markdown("请上传一个PDF文件，然后开始就文件内容进行提问。")

    # 使用Gradio State来存储会话中需要持久化的对象（这里是RAG链）
    # 这样就无需在每次提问时都重新创建RAG链
    rag_chain_state = gr.State()

    with gr.Row():
        with gr.Column(scale=1):
            upload_button = gr.UploadButton(
                "点击上传PDF文件",
                file_types=[".pdf"],
                file_count="single"
            )
            process_status = gr.Textbox(
                label="文件处理状态",
                value="请先上传一个PDF文件",
                interactive=False
            )

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="聊天窗口")
            msg_input = gr.Textbox(label="输入你的问题...", interactive=False)
            clear_button = gr.ClearButton([msg_input, chatbot], value="清空聊天记录")

    # --- 设定组件之间的交互逻辑 ---

    # 当用户上传文件后，触发process_file函数
    # 函数的输出会更新 rag_chain_state 和 process_status
    upload_button.upload(
        process_file,
        inputs=[upload_button],
        outputs=[rag_chain_state, process_status]
    )

    # 当用户在输入框提交问题时，触发chat_with_doc函数
    # `chat_with_doc`会接收消息、历史记录和我们存储的RAG链状态
    msg_input.submit(
        chat_with_doc,
        inputs=[msg_input, chatbot, rag_chain_state],
        outputs=[msg_input, chatbot]
    )


# --- 5. 启动应用 ---
if __name__ == "__main__":
    demo.launch(share=True) # share=True 会生成一个公开链接，方便分享