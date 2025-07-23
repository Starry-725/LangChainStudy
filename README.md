<!--
 * @Descripttion: 说明
 * @version: V1.0
 * @Author: StarryLei
 * @Date: 2025-07-17 22:46:27
 * @LastEditors: Starry 1018485883@qq.com
 * @LastEditTime: 2025-07-23 17:19:58
-->
# 说明
** 这里只是存储一些个人学习Langchain的一些基本代码，方便在不同机器上同步代码内容，不做他用。 **

# 运行环境
pip install -r requirements.txt

# 环境变量
需要在主目录下创建一个.env文件存放一些API_KEY

## 1_load_LLM.py
1. 使用Langchain接入各类大语言模型，实验对象有deepseek、openai这样的官方通用接口。
2. 火山引擎这样官方没有提供init_chat_model方法的接口，Qwen则是需要通过DashScope去加载模型，这些与传统的init_chat_model方法有一定的区别。
3. 本地模型则是通过vLLM或者Ollma去接入。

# 2_construct_chains.py
1. 构造一个chain去加入提示模板、构建结构化输出、结构化输出解析。
2. 通过格式化解析结果构建符合链，可以将不同的链串联起来。

# 3_chat_robot.py
构造一个支持多轮对话的机器人，追加历史记录的方式展示两种：  
1. 用messages_list去传递,将问答历史用append的方式追加到list中
2. 用RunnableWithMessageHistory创建一个chain,将简单的对话链包装成带记忆的链。调用方法去根据session获得ChatMessageHistory。
3. 基于gradio界面化的实现了第二种方法。运行命令： python 3_chat_robot.py

# 4_my_tool.py
1. 编写一个工具，用于查询某个地方的天气
2. langchain自带的浏览器检索工具TavilySearch的使用

# 5_agent_sql_db2excel.py
编写了一个自己构建的Agent：将用户输入的自然语言转化为SQL语言，并将查询结果写入Excel文件中。

# 6_agent_debate.py
创建了三个Agent，用户只需要提供一个辩论题目，便可自动产生一场辩论赛。
1. 裁判Agent：（1）结合辩题生成两个对立观点。（2）根据三个维度为正反方辩手打分。（3）根据辩论全部过程对辩论胜负作出裁决。
2. 正反方辩手Agent：（1）根据己方的论点以及辩论历史进行网页检索，搜寻己方观点的有力论据。（2）根据对方辩手的论据和自己的检索论据，作出己方观点的论证。
3. 正反方辩手Agent中含有TavilySearch网页搜索工具，辩手通过这个进行文献和论证搜索。
4. 需要一个记录工具，用于将整个辩论过程记录为Markdown格式的日志。