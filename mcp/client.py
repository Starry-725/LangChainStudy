'''
Author: StarryLei 1018485883@qq.com
Date: 2025-07-30 23:18:34
LastEditors: StarryLei 1018485883@qq.com
LastEditTime: 2025-07-30 23:46:03
FilePath: \LangChainStudy\mcp-client\client.py
Description: 
1.创建MCP客户端
2.获取工具列表
3.调用工具
'''

# client.py

import subprocess
import json
import uuid
import time

class MCPClient:
    """
    一个用于与基于stdio的MCP服务器交互的客户端。
    """
    def __init__(self, server_script_path="mcp_server.py"):
        self.server_script_path = server_script_path
        self.process = None

    def start_server(self):
        """启动服务器子进程。"""
        print("正在启动MCP服务器子进程...")
        self.process = subprocess.Popen(
            ["python", "-u", self.server_script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, # 捕获服务器的错误输出以供调试
            text=True, # 以文本模式处理流
            encoding='utf-8'
        )
        # 等待一小段时间确保服务器完全启动
        time.sleep(2) 
        print("服务器已启动。")

    def stop_server(self):
        """停止服务器子进程。"""
        if self.process:
            print("正在停止MCP服务器...")
            self.process.terminate()
            self.process.wait()
            print("服务器已停止。")
            # 打印服务器的任何错误日志
            stderr_output = self.process.stderr.read()
            if stderr_output:
                print("\n--- 服务器错误日志 ---")
                print(stderr_output)
                print("----------------------")


    def send_command(self, command: str) -> dict:
        """向服务器发送一个命令并获取响应。"""
        if not self.process:
            raise ConnectionError("服务器未启动。请先调用 start_server()。")

        request_id = str(uuid.uuid4())
        request = {
            "id": request_id,
            "command": command
        }
        
        json_request = json.dumps(request)
        print(f"\n[客户端 -> 服务器] 发送: {json_request}")

        try:
            # 写入服务器的stdin
            self.process.stdin.write(json_request + '\n')
            self.process.stdin.flush()

            # 从服务器的stdout读取响应
            response_line = self.process.stdout.readline()
            if not response_line:
                raise ConnectionAbortedError("与服务器的连接已断开。")

            response = json.loads(response_line)
            print(f"[服务器 -> 客户端] 收到: {json.dumps(response, indent=2, ensure_ascii=False)}")
            return response
        
        except (BrokenPipeError, ConnectionAbortedError) as e:
            print(f"错误：与服务器的连接中断。 {e}")
            self.stop_server()
            return {"status": "error", "payload": str(e)}


if __name__ == "__main__":
    client = MCPClient()
    try:
        client.start_server()
        
        # --- 运行一些测试命令 ---
        client.send_command("你好，你是谁？")
        
        # 使用工具的命令
        client.send_command("帮我看看 'project_alpha' 文件夹里有什么。")
        
        # 触发安全限制的命令
        client.send_command("我想看看沙箱外面的 '../' 目录。")
        
        # 发送一个格式错误的命令 (不是JSON) - 这将由服务器的顶级异常处理捕获
        # print("\n[客户端 -> 服务器] 发送一个非JSON的无效请求...")
        # client.process.stdin.write("this is not json\n")
        # client.process.stdin.flush()
        # invalid_response = json.loads(client.process.stdout.readline())
        # print(f"[服务器 -> 客户端] 收到: {json.dumps(invalid_response, indent=2)}")

    finally:
        # 确保无论发生什么，服务器都会被关闭
        client.stop_server()