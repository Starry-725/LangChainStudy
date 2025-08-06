
# sse_mcp_server.py

import uvicorn
import json
import os
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

# --- LangChain 和工具相关模块 ---
from langchain.tools import tool
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain import hub
from dotenv import load_dotenv

# --- 1. 环境准备与安全沙箱 ---
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("请在.env文件中设置OPENAI_API_KEY")

# 同样使用我们之前定义的安全文件系统工具
SAFE_BASE_DIRECTORY = os.path.abspath("./sandbox")

@tool
def list_directory_contents(directory_path: str) -> str:
    """查看指定子目录中的文件和文件夹列表。只能访问预设的安全沙箱。"""
    try:
        clean_path = os.path.normpath(directory_path)
        if '..' in clean_path.split(os.sep): raise PermissionError("路径穿越尝试被拒绝。")
        full_path = os.path.join(SAFE_BASE_DIRECTORY, clean_path)
        if not os.path.realpath(full_path).startswith(SAFE_BASE_DIRECTORY): raise PermissionError("请求路径在安全沙箱之外。")
        if not os.path.isdir(full_path): return f"错误：'{directory_path}' 不是一个文件夹。"
        contents = [{"name": e.name, "type": "dir" if e.is_dir() else "file"} for e in os.scandir(full_path)]
        return json.dumps(contents, indent=2)
    except Exception as e:
        return f"错误: {e}"

# --- 2. FastAPI应用和LangChain Agent初始化 ---
app = FastAPI()

# Agent只在服务器启动时构建一次，以提高效率
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0, streaming=True)
tools = [list_directory_contents]
prompt = hub.pull("hwchase17/openai-tools-agent")
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) # Verbose设为True可以在服务器端看到详细日志

# --- 3. SSE核心逻辑 ---

async def stream_generator(command: str):
    """
    这是一个异步生成器，它调用Agent的astream，处理每个事件块，
    并将其格式化为SSE消息后yield出去。
    """
    try:
        # 使用异步流 astream
        async for chunk in agent_executor.astream({"input": command}):
            event_type = ""
            data = {}

            # 解析LangChain流的不同事件类型
            if "actions" in chunk:
                action = chunk["actions"][0]
                event_type = "tool_call"
                data = {"tool": action.tool, "tool_input": action.tool_input, "log": action.log}
            elif "steps" in chunk:
                step = chunk["steps"][0]
                event_type = "tool_output"
                data = {"tool_output": step.observation}
            elif "output" in chunk:
                event_type = "final_answer"
                data = {"answer": chunk["output"]}
            else:
                # 忽略其他中间事件或继续处理
                continue
            
            # 格式化为SSE消息
            # event: <event_name>
            # data: <json_string>
            # \n\n
            sse_message = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
            yield sse_message
            await asyncio.sleep(0.1) # 短暂休眠，防止CPU占用过高

    except Exception as e:
        # 如果在流处理中发生错误，发送一个错误事件
        error_data = {"error": str(e)}
        sse_message = f"event: error\ndata: {json.dumps(error_data)}\n\n"
        yield sse_message

@app.get("/mcp-stream")
async def mcp_stream_endpoint(command: str):
    """
    接收客户端命令，返回一个Server-Sent Events (SSE)流。
    """
    # StreamingResponse是FastAPI用于处理流式响应的关键
    return StreamingResponse(stream_generator(command), media_type="text/event-stream")


# --- 4. 运行服务器 ---
if __name__ == "__main__":
    print("启动FastAPI服务器，访问 http://127.0.0.1:8000")
    print("使用客户端或浏览器访问端点，例如: http://127.0.0.1:8000/mcp-stream?command=list+project_alpha")
    uvicorn.run(app, host="127.0.0.1", port=8000)

