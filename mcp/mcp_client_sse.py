# sse_client.py

import requests
import json
import urllib.parse
from sseclient import SSEClient

class SSE_MCPClient:
    """
    一个用于与基于SSE的MCP服务器交互的客户端。
    """
    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url

    def stream_command(self, command: str):
        """
        连接到服务器的SSE端点，并实时打印事件。
        """
        # 对命令进行URL编码，以安全地作为查询参数传递
        encoded_command = urllib.parse.quote_plus(command)
        url = f"{self.base_url}/mcp-stream?command={encoded_command}"

        print(f"--- 正在连接到 {url} ---")
        
        try:
            # 使用requests库以流模式连接
            response = requests.get(url, stream=True)
            response.raise_for_status() # 如果HTTP状态码不是2xx，则抛出异常

            client = SSEClient(response)

            print("--- 开始接收事件流 ---\n")
            for event in client.events():
                print(f"EVENT: [{event.event}]")
                
                # 美化打印JSON数据
                try:
                    data_obj = json.loads(event.data)
                    pretty_data = json.dumps(data_obj, indent=2, ensure_ascii=False)
                    print(f"DATA: \n{pretty_data}\n")
                except json.JSONDecodeError:
                    print(f"DATA: {event.data}\n")

                # 如果是最终答案或错误，可以考虑退出循环
                if event.event in ["final_answer", "error"]:
                    print("--- 事件流结束 ---")
                    break

        except requests.exceptions.RequestException as e:
            print(f"\n错误：无法连接到服务器。请确保服务器正在运行。")
            print(f"详细信息: {e}")
        except Exception as e:
            print(f"\n处理事件流时发生未知错误: {e}")


if __name__ == "__main__":
    client = SSE_MCPClient()
    
    # --- 运行一个测试命令 ---
    # 这个命令会触发Agent调用工具
    user_command = "帮我看看 project_alpha 文件夹里有什么，然后告诉我里面最重要的文件是什么？"
    client.stream_command(user_command)