# 说明
** 这里只是存储一些个人学习Langchain的一些基本代码，方便在不同机器上同步代码内容，不做他用。 **

# 运行环境
pip install -r requirements.txt

# 环境变量
需要在主目录下创建一个.env文件存放一些API_KEY

## 1_load_LLM.py
1.使用Langchain接入各类大语言模型，实验对象有deepseek、openai这样的官方通用接口。
2.火山引擎这样官方没有提供init_chat_model方法的接口，Qwen则是需要通过DashScope去加载模型，这些与传统的init_chat_model方法有一定的区别。
3.本地模型则是通过vLLM或者Ollma去接入。

# 2_construct_chains.py
1.构造一个chain去加入提示模板、构建结构化输出、结构化输出解析。
2.通过格式化解析结果构建符合链，可以将不同的链串联起来。

# 3_chat_robot.py
构造一个支持多轮对话的机器人，追加历史记录的方式展示两种：
1.用messages_list去传递,将问答历史用append的方式追加到list中
2.用RunnableWithMessageHistory创建一个chain,将简单的对话链包装成带记忆的链。调用方法去根据session获得ChatMessageHistory。
3.基于gradio界面化的实现了第二种方法。
