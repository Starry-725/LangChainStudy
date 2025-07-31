'''
Author: StarryLei 1018485883@qq.com
Date: 2025-07-30 22:59:27
LastEditors: StarryLei 1018485883@qq.com
LastEditTime: 2025-07-31 22:45:15
FilePath: \LangChainStudy\mcp-client\server.py
Description: 创建一个简单的ＭＣＰ服务器示例
'''

from fastmcp import FastMCP

mcp = FastMCP()

@mcp.tool()
def get_welcome_message(name: str) -> str:
    """
    返回欢迎消息
    :param name: 用户名
    :return: 欢迎消息
    """
    return f"Welcome to the MCP Server, {name}!"

if __name__ == "__main__":
    # 启动MCP服务器
    mcp.run(transport="stdio")