'''
Author: StarryLei 1018485883@qq.com
Date: 2025-07-30 22:59:27
LastEditors: StarryLei 1018485883@qq.com
LastEditTime: 2025-07-30 23:35:15
FilePath: \LangChainStudy\mcp-client\server.py
Description: 创建一个简单的ＭＣＰ服务器示例
'''

from fastmcp import FastMCP

mcp = FastMCP()

@mcp.tool()
def get_welcome_message(name: str) -> str:
    return f"Welcome to the MCP Server, {name}!"

if __name__ == "__main__":
    # 启动MCP服务器
    mcp.run()