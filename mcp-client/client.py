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

import asyncio
from fastmcp import Client


async def main():
    # 创建MCP客户端
    client = Client('server.py')

    async with client:
        # 获取工具列表
        tools = await client.list_tools()
        print("可用工具列表:", tools)

        # 调用工具
        if tools:
            tool = tools[0]  # 选择第一个工具
            result = await client.call_tool(tool.name, arguments={"name": "StarryLei"})
            print(f"调用工具 '{tool.name}' 的结果: {result}")
        else:
            print("没有可用的工具")


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())