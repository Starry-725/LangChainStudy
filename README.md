# 说明
** 这里只是存储一些个人学习Langchain的一些基本代码，方便在不同机器上同步代码内容，不做他用。 **

# 运行环境
pip install -r requirements.txt

## 1_load_LLM.py
使用Langchain加载各类大语言模型，实验对象有deepseek、openai这样的官方通用接口。有火山引擎这样官方没有提供init_chat_model方法的接口，Qwen则是需要通过DashScope去加载模型，这些与传统的init_chat_model方法有一定的区别。

# 2_construct_chains.py

