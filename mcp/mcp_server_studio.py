'''
Author: StarryLei 1018485883@qq.com
Date: 2025-07-30 22:59:27
LastEditors: StarryLei 1018485883@qq.com
LastEditTime: 2025-07-31 22:45:15
FilePath: \LangChainStudy\mcp-client\server.py
Description: 创建一个简单的ＭＣＰ服务器示例
'''

# mcp_server.py

import sys
import json
import os
import logging
from datetime import datetime

# --- LangChain 和工具相关模块 ---
from langchain.tools import tool
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain import hub
from dotenv import load_dotenv

# --- 1. 配置日志记录 ---
# 将日志输出到文件，保持stdout干净，用于IPC通信
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='mcp_server.log',
    filemode='a'
)

# --- 2. 环境准备与安全沙箱 ---
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("请在.env文件中设置OPENAI_API_KEY")

# 复制我们之前创建的安全文件系统工具
SAFE_BASE_DIRECTORY = os.path.abspath("./sandbox")

class DirectoryPathInput(BaseModel):
    directory_path: str = Field(description="要查看的目录的相对路径。")

@tool(args_schema=DirectoryPathInput)
def list_directory_contents(directory_path: str) -> str:
    """查看指定子目录中的文件和文件夹列表。只能访问预设的安全沙箱。"""
    try:
        clean_path = os.path.normpath(directory_path)
        if '..' in clean_path.split(os.sep):
            raise PermissionError("检测到路径穿越尝试 ('../')，操作被拒绝。")
        full_path = os.path.join(SAFE_BASE_DIRECTORY, clean_path)
        if os.path.commonprefix([os.path.realpath(full_path), SAFE_BASE_DIRECTORY]) != SAFE_BASE_DIRECTORY:
            raise PermissionError("访问被拒绝：请求的路径在安全沙箱之外。")
        if not os.path.exists(full_path) or not os.path.isdir(full_path):
            return f"错误：目录 '{directory_path}' 不存在或不是一个文件夹。"
        
        contents = [{
            "name": entry.name,
            "type": "directory" if entry.is_dir() else "file",
            "size_bytes": entry.stat().st_size
        } for entry in os.scandir(full_path)]

        return json.dumps(contents, indent=2) if contents else f"目录 '{directory_path}' 是空的。"
    except PermissionError as e:
        return f"安全错误: {e}"
    except Exception as e:
        return f"处理请求时发生未知错误: {e}"

# --- 3. 构建核心Agent ---
def build_agent():
    """构建并返回一个配置好的LangChain Agent Executor。"""
    logging.info("正在构建Agent...")
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
    tools = [list_directory_contents]
    prompt = hub.pull("hwchase17/openai-tools-agent")
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False) # 在服务器模式下通常关闭verbose
    logging.info("Agent构建完成。")
    return agent_executor

# --- 4. 主服务器循环 ---
def main_loop(agent_executor):
    """
    监听stdin，处理请求，并通过stdout响应。
    """
    logging.info("MCP服务器已启动，正在监听stdin...")
    while True:
        try:
            # 从标准输入读取一行
            line = sys.stdin.readline()

            # 如果读取到空行，意味着客户端可能已关闭连接，退出循环
            if not line:
                logging.info("检测到空的输入流，服务器正在关闭。")
                break

            # 解析JSON请求
            request = json.loads(line)
            request_id = request.get("id", "no-id")
            command = request.get("command")
            logging.info(f"收到请求 [ID: {request_id}]: {command}")

            if not command:
                raise ValueError("请求中缺少 'command' 字段。")

            # 使用Agent处理命令
            try:
                agent_response = agent_executor.invoke({"input": command})
                payload = agent_response.get("output", "Agent没有提供输出。")
                response = {"id": request_id, "status": "success", "payload": payload}
            except Exception as e:
                logging.error(f"Agent执行时出错 [ID: {request_id}]: {e}", exc_info=True)
                response = {"id": request_id, "status": "error", "payload": str(e)}

        except json.JSONDecodeError:
            logging.error(f"无法解析收到的行: {line.strip()}")
            response = {"id": "unknown", "status": "error", "payload": "无效的JSON请求。"}
        except Exception as e:
            logging.error(f"处理请求时发生未知错误: {e}", exc_info=True)
            response = {"id": "unknown", "status": "error", "payload": str(e)}

        # 将JSON响应写入标准输出
        json_response = json.dumps(response)
        sys.stdout.write(json_response + '\n')
        # **关键**: 刷新输出缓冲区，确保客户端立即收到消息
        sys.stdout.flush()
        logging.info(f"已发送响应 [ID: {response.get('id', 'unknown')}]")


if __name__ == "__main__":
    agent_executor = build_agent()
    main_loop(agent_executor)