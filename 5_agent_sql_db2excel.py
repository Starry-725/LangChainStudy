import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import (
    create_engine,
    text,
    Connection
)
from typing import List, Dict, Any
from langchain.tools import tool
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_tools_agent, AgentExecutor

# --- 准备工作：设置OpenAI API Key ---
# 建议通过环境变量设置，避免硬编码
# from getpass import getpass
# os.environ["OPENAI_API_KEY"] = getpass("Enter your OpenAI API key: ")
load_dotenv(override=True)

if not os.environ.get("ARK_API_KEY"):
    raise ValueError("请设置 ARK_API_KEY 环境变量")

ARK_API_KEY = os.getenv("ARK_API_KEY")
# --- 第二步：创建并连接到模拟数据库 ---
# 使用内存SQLite数据库，方便演示。每次运行程序时都会重新创建。
# `file:memdb1?mode=memory&cache=shared` 允许多次连接到同一个内存数据库。 [3]
# `check_same_thread=False` 是SQLite在多线程环境下的一个设置。
engine = create_engine("sqlite:///:memory:", echo=False)

def setup_database(conn: Connection):
    """创建表并插入数据"""
    conn.execute(text("""
    CREATE TABLE employees (
        id INTEGER PRIMARY KEY,
        name VARCHAR(50),
        department VARCHAR(50),
        salary INTEGER,
        age INTEGER
    )
    """))
    conn.execute(text("""
    INSERT INTO employees (id, name, department, salary, age) VALUES
    (1, '张三', '技术部', 8000, 35),
    (2, '李四', '技术部', 9500, 28),
    (3, '王五', '市场部', 6000, 42),
    (4, '赵六', '市场部', 7500, 31),
    (5, '孙七', '人事部', 5500, 25)
    """))
    conn.commit()

# 执行数据库初始化
with engine.connect() as connection:
    setup_database(connection)

print("数据库已成功创建并填充数据。")

# --- 第一个工具：数据库查询 ---
@tool
def query_database(sql_query: str) -> List[Dict[str, Any]]:
    """
    执行一个SQL查询语句来从员工数据库中获取信息。
    数据库包含一张名为 'employees' 的表，
    字段有: id, name, department, salary, age。
    请使用标准的SQL语法。
    例如: SELECT * FROM employees WHERE department = '技术部'
    """
    with engine.connect() as connection:
        result = connection.execute(text(sql_query))
        # 将结果转换为字典列表，方便处理
        data = [row._asdict() for row in result.fetchall()]
        return data

# --- 第二个工具：写入Excel文件 ---
# 使用Pydantic定义输入模型，确保Agent提供正确的参数
class ExcelInput(BaseModel):
    data: List[Dict[str, Any]] = Field(description="需要写入Excel的数据，必须是字典组成的列表")
    filename: str = Field(description="要保存的Excel文件名，例如 'output.xlsx'")

@tool(args_schema=ExcelInput)
def write_to_excel(data: List[Dict[str, Any]], filename: str) -> str:
    """
    将一个字典列表（List of Dictionaries）写入到指定的Excel文件中。
    这个工具在你已经获取到数据之后使用。
    """
    if not data:
        return "数据为空，无法创建Excel文件。"
    df = pd.DataFrame(data)
    try:
        df.to_excel(filename, index=False)
        return f"数据已成功写入到文件 '{filename}'。"
    except Exception as e:
        return f"写入Excel文件时发生错误: {e}"

# 将所有工具放入一个列表
tools = [query_database, write_to_excel]


# 1. 初始化LLM
# temperature=0 表示我们希望模型有更稳定、更具确定性的输出
llm = ChatOpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=ARK_API_KEY,
        model="ep-m-20250719172710-9zfxx",
        streaming=True
    )

# 2. 获取预设的Agent提示模板
# 这个模板指导LLM如何进行思考、使用工具并最终给出答案
prompt = hub.pull("hwchase17/openai-tools-agent")

# 3. 创建Agent
# 将LLM、工具和提示模板结合起来
agent = create_openai_tools_agent(llm, tools, prompt)

# 4. 创建Agent执行器
# AgentExecutor负责循环运行Agent，直到任务完成或达到最大迭代次数
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# 提出一个需要组合使用两个工具的复杂请求
user_prompt = "请帮我查询所有在'技术部'并且薪水高于9000元的员工信息，然后将结果保存到名为'tech_high_salary.xlsx'的Excel文件中。"

# 运行Agent
response = agent_executor.invoke({
    "input": user_prompt
})

print("\n--- Agent最终回复 ---")
print(response["output"])

# --- 验证文件是否生成 ---
try:
    if os.path.exists('tech_high_salary.xlsx'):
        print("\n--- 验证Excel文件内容 ---")
        df_read = pd.read_excel('tech_high_salary.xlsx')
        print(df_read)
    else:
        print("\n文件 'tech_high_salary.xlsx' 未找到。")
except Exception as e:
    print(f"\n读取Excel文件时出错: {e}")