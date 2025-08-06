'''
Descripttion: 辩论Agent，具备搜索和记录功能
version: V2.0
Author: StarryLei & Gemini
Date: 2025-07-22 00:06:16
LastEditors: Starry 1018485883@qq.com
LastEditTime: 2025-08-05 14:49:18
'''
import json
import time
import os
from dotenv import load_dotenv
from typing import Dict, Any

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
# NEW: 导入Agent和工具相关模块
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool
from langchain import hub


# --- 环境准备 ---
load_dotenv(override=True)

# 检查所有需要的API密钥
if not os.environ.get("ARK_API_KEY"):
    raise ValueError("请设置 ARK_API_KEY 环境变量")
if not os.environ.get("TAVILY_API_KEY"):
    raise ValueError("请设置 TAVILY_API_KEY 环境变量 (用于搜索功能)")

ARK_API_KEY = os.getenv("ARK_API_KEY")

# --- 定义Agent可以使用的工具 ---

# NEW: 1. 搜索工具
# TavilySearchResults是一个封装好的、易于使用的搜索工具
search_tool = TavilySearchResults(max_results=3)
search_tool.name = "web_search" # 给工具一个简单明确的名称

# NEW: 2. Markdown文件写入工具
@tool
def save_to_markdown(filename: str, content: str) -> str:
    """
    将指定的文本内容追加到Markdown文件的末尾。
    用于记录辩论的每一个步骤。
    """
    try:
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(content + "\n\n")
        return f"内容已成功记录到 {filename}"
    except Exception as e:
        return f"写入文件时发生错误: {e}"

# --- 更新提示模板 ---

# 裁判的提示保持不变，因为其职责没有改变
referee_template = """
你的角色是一位经验丰富、绝对中立的辩论赛裁判。你的任务是基于逻辑、证据和说服力来评判辩论。
# 辩论总议题: {topic}
# 你的职责:
1.  **生成立场**: 根据总议题，为正反双方生成清晰、明确、有可辩论空间的对立立场。请以JSON格式返回，键为 "pro_stance" 和 "con_stance"。
2.  **单轮评分**: 在每一轮辩论中，你需要对刚刚发言的一方进行评分。评分标准如下：
    - **逻辑清晰度 (1-5分)**: 论点是否条理清晰，推理过程是否严谨。
    - **论据支撑力 (1-5分)**: 是否使用了有效的事实、数据或例子来支持观点。
    - **说服力与表达 (1-5分)**: 语言是否流畅，表达是否有力，是否能打动人心。
    请在听完一方的陈述后，以JSON格式返回评分，包含 "score" (总分15分制) 和 "reasoning" (评分理由)。
3.  **最终裁决**: 在所有轮次结束后，根据双方总分，宣布获胜方并提供一段总结性陈词。
# 辩论历史:
{history}
---
现在，请履行你的职责。
{task}
"""
referee_prompt = ChatPromptTemplate.from_template(referee_template)


# MODIFIED: 辩手的提示，增加了使用工具的指令
# 注意：我们不再直接使用这个模板，而是将其中的核心指令思想融入到Agent的提示中
# 这里保留它用于理解，实际使用的是下面的 `hub.pull` 的提示
debater_agent_prompt_template = """
你是一位顶级的辩论选手。你的风格是逻辑严密，言辞犀利，同时富有说服力。

# 辩论总议题: {topic}
# 你的立场: {stance}

# 你的任务:
1.  **思考与搜集**: 首先，思考你的论点。然后，你**必须**使用 `web_search` 工具来查找支持你观点的数据、事实、案例或权威引述。好的论据是获胜的关键。
2.  **构建论点**: 结合你搜集到的信息和你的立场，构建一段有力的辩词。
3.  **反驳对手**: 如果辩论历史中有对手的观点，你需要有针对性地进行反驳。
4.  **保持角色**: 始终坚定地捍卫你的立场。

# 辩论历史:
{history}

---
现在轮到你发言了。开始思考并使用工具，最终输出你的辩词。
"""

# --- 初始化LLM和Agent ---

# LLM模型实例
referee_llm = ChatOpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=ARK_API_KEY,
    model="ep-m-20250719172710-9zfxx",
    # streaming=True, # 在Agent模式下，流式输出处理更复杂，暂时关闭以简化
)

pro_llm = ChatOpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=ARK_API_KEY,
    model="ep-m-20250723164632-ctnnr",
    # streaming=True, # 在Agent模式下，流式输出处理更复杂，暂时关闭以简化
)

con_llm = ChatOpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=ARK_API_KEY,
    model="ep-m-20250411184749-5qknb",
    # streaming=True, # 在Agent模式下，流式输出处理更复杂，暂时关闭以简化
)


# 裁判Agent保持不变，因为它不需要使用工具
referee_agent = referee_prompt | referee_llm

# NEW: 创建能够使用搜索工具的辩手Agent
# 辩手需要一个专门的Agent提示，我们从LangChain Hub拉取一个标准模板
# 这个模板知道如何指示LLM进行思考、调用工具、观察结果
agent_prompt = hub.pull("hwchase17/openai-tools-agent")

# 为正反方辩手分别创建Agent执行器
# 他们共享同一个LLM和提示模板，但可以使用自己的工具集（虽然这里工具集也相同）
debater_tools = [search_tool]
pro_debater_agent = create_openai_tools_agent(pro_llm, debater_tools, agent_prompt)
con_debater_agent = create_openai_tools_agent(con_llm, debater_tools, agent_prompt)

pro_agent_executor = AgentExecutor(agent=pro_debater_agent, tools=debater_tools, verbose=True)
con_agent_executor = AgentExecutor(agent=con_debater_agent, tools=debater_tools, verbose=True)


# MODIFIED: 增强的DebateManager
class DebateManager:
    def __init__(self, topic, rounds=4):
        self.topic = topic
        self.rounds = rounds
        self.history = ""
        self.scores = {"pro": 0, "con": 0}
        self.pro_stance = ""
        self.con_stance = ""
        # NEW: 初始化Markdown文件名和写入工具
        self.markdown_filename = f"debate_log_{time.strftime('%Y%m%d_%H%M%S')}.md"
        self.markdown_writer = save_to_markdown

    def _call_referee(self, task: str) -> Dict[str, Any]:
        """调用裁判并解析其JSON输出"""
        response = referee_agent.invoke({
            "topic": self.topic,
            "history": self.history,
            "task": task
        })
        try:
            cleaned_response = response.content.strip().replace("```json", "").replace("```", "")
            return json.loads(cleaned_response)
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"裁判返回了非JSON格式的回应或错误: {response}")
            return None

    def _record(self, content: str):
        """调用工具记录内容到Markdown文件"""
        print(f"\n[记录员] 正在将内容写入 {self.markdown_filename}...")
        self.markdown_writer.invoke({
            "filename": self.markdown_filename,
            "content": content
        })

    def setup_debate(self):
        """让裁判生成正反方立场，并记录"""
        print("--- 辩论准备阶段 ---")
        print(f"裁判正在根据议题 '{self.topic}' 生成双方立场...")
        
        task = "请为本次辩论生成正反双方的立场。"
        stances = self._call_referee(task)
        
        if stances and "pro_stance" in stances and "con_stance" in stances:
            self.pro_stance = stances["pro_stance"]
            self.con_stance = stances["con_stance"]
            
            # 构造初始历史和记录内容
            header = f"# 辩论议题：{self.topic}\n\n"
            pro_stance_md = f"**正方立场**: {self.pro_stance}\n"
            con_stance_md = f"**反方立场**: {self.con_stance}\n"
            
            self.history = header + pro_stance_md + con_stance_md
            self._record(self.history) # 首次记录

            print(f"正方立场: {self.pro_stance}")
            print(f"反方立场: {self.con_stance}")
        else:
            raise ValueError("无法从裁判处获得有效的辩题。")

    # MODIFIED: 增强的DebateManager中的run_round方法
    def run_round(self, round_num: int):
        """
        运行一轮辩论，并记录所有步骤。
        此版本通过构建更具体的输入提示来指导Agent进行相关搜索。
        """
        round_header = f"## 第 {round_num} 轮辩论"
        print(f"\n--- {round_header} ---")
        self._record(round_header)

        # --- 正方发言 ---
        print("\n[正方准备发言...]")
        
        # MODIFIED: 创建一个高度具体的、包含所有上下文的输入提示
        # 这是解决无关搜索问题的关键！我们明确告诉Agent它的角色、议题、立场，并给出搜索关键词的示例。
        pro_input_prompt = f"""
        现在轮到你作为正方发言。
        议题: "{self.topic}"
        你的立场是: "{self.pro_stance}"

        你的任务:
        1.  **制定搜索策略**: 基于你的立场，思考需要哪些证据来支撑。
        2.  **执行搜索**: 调用 `web_search` 工具，使用与你的立场和议题**高度相关**的关键词进行搜索。例如，针对你的立场，可以搜索类似“社交媒体对青少年心理健康的正面影响研究”、“青少年如何通过社交媒体建立社区”等。
        3.  **构建论点**: 结合搜索到的信息和已有的辩论历史，构建你的发言。如果历史中有对手的观点，请进行反驳。

        请开始你的思考和行动。
        """
        
        # MODIFIED: 调用Agent Executor时，传入这个全新的、信息丰富的input
        # 我们不再需要单独传递 topic 和 stance，因为它们已经包含在 input 里了。
        pro_response = pro_agent_executor.invoke({
            "input": pro_input_prompt,
            "history": self.history
        })
        pro_argument = pro_response['output']
        print(f"\n[正方辩手]:\n{pro_argument}")
        pro_record = f"### 正方发言\n\n{pro_argument}"
        self._record(pro_record)
        self.history += f"\n第{round_num}轮 - 正方: {pro_argument}\n"
        
        # 裁判为正方评分 (这部分逻辑不变)
        pro_score_result = self._call_referee("请为刚才正方辩手的陈述评分。")
        print(pro_score_result)
        if pro_score_result:
            score = pro_score_result.get('score', 0)
            if isinstance(score, dict):
                score = score['逻辑清晰度'] + score["论据支撑力"] + score["说服力与表达"]
            else:
                pass
            reasoning = pro_score_result.get('reasoning', '无理由')
            self.scores["pro"] += score
            pro_score_md = f"#### 裁判点评 (正方)\n\n- **得分**: {score}/15\n- **理由**: {reasoning}\n"
            print(f"\n[裁判点评-正方]:\n分数: {score}/15\n理由: {reasoning}\n")
            self._record(pro_score_md)
            self.history += f"裁判评分(正方): {score}/15, 理由: {reasoning}\n"

        time.sleep(2) # 稍微暂停，让流程更清晰

        # --- 反方发言 ---
        print("\n[反方准备发言...]")
        
        # MODIFIED: 为反方也创建一个高度具体的输入提示
        con_input_prompt = f"""
        现在轮到你作为反方发言，进行反驳和陈述。
        议题: "{self.topic}"
        你的立场是: "{self.con_stance}"

        你的任务:
        1.  **分析对手**: 阅读辩论历史中正方的最新观点。
        2.  **制定搜索策略**: 思考如何反驳对方，以及需要哪些证据来加强你的立场。
        3.  **执行搜索**: 调用 `web_search` 工具，使用与你的立场和议题**高度相关**的关键词进行搜索。例如，搜索类似“社交媒体与青少年抑郁症的关联”、“网络霸凌的统计数据”等。
        4.  **构建论点**: 结合搜索到的信息和辩论历史，有力地反驳对方并阐述你的观点。

        请开始你的思考和行动。
        """
        
        con_response = con_agent_executor.invoke({
            "input": con_input_prompt,
            "history": self.history
        })
        con_argument = con_response['output']
        print(f"\n[反方辩手]:\n{con_argument}")
        con_record = f"### 反方发言\n\n{con_argument}"
        self._record(con_record)
        self.history += f"\n第{round_num}轮 - 反方: {con_argument}\n"

        # 裁判为反方评分 (这部分逻辑不变)
        con_score_result = self._call_referee("请为刚才反方辩手的陈述评分。")
        if con_score_result:
            score = con_score_result.get('score', 0)
            if isinstance(score, dict):
                score = score['逻辑清晰度'] + score["论据支撑力"] + score["说服力与表达"]
            else:
                pass
            reasoning = con_score_result.get('reasoning', '无理由')
            self.scores["con"] += score
            con_score_md = f"#### 裁判点评 (反方)\n\n- **得分**: {score}/15\n- **理由**: {reasoning}\n"
            print(f"\n[裁判点评-反方]:\n分数: {score}/15\n理由: {reasoning}\n")
            self._record(con_score_md)
            self.history += f"裁判评分(反方): {score}/15, 理由: {reasoning}\n"

    def announce_winner(self):
        """宣布最终结果并记录"""
        final_header = "## 辩论结束"
        print(f"\n--- {final_header} ---")
        
        final_scores_text = f"最终得分 -> 正方: {self.scores['pro']} | 反方: {self.scores['con']}"
        print(final_scores_text)

        if self.scores['pro'] > self.scores['con']:
            winner = "正方"
        elif self.scores['con'] > self.scores['pro']:
            winner = "反方"
        else:
            winner = "平局"
        
        final_task = f"辩论已结束，双方总分分别是正方 {self.scores['pro']} 和反方 {self.scores['con']}。请宣布 '{winner}' 获胜，并发表一段最终的总结陈词。"
        summary_response = referee_agent.invoke({
            "topic": self.topic,
            "history": self.history,
            "task": final_task
        })
        summary_content = summary_response.content
        print("\n[裁判最终裁决]:")
        print(summary_content)

        # 记录最终结果
        final_record = f"{final_header}\n\n**{final_scores_text}**\n\n### 裁判最终裁决\n\n{summary_content}"
        self._record(final_record)
        print(f"\n完整辩论过程已记录在文件: {self.markdown_filename}")

    def run_debate(self):
        """完整运行整个辩论"""
        self.setup_debate()
        for i in range(1, self.rounds + 1):
            self.run_round(i)
            time.sleep(1) # 每轮之间稍作停顿
        self.announce_winner()


if __name__ == "__main__":
    debate_topic = "社交媒体对青少年的心理健康是积极影响大于消极影响，还是消极影响大于积极影响？"
    
    # 为了演示，我们只进行1轮，你可以增加轮数
    manager = DebateManager(topic=debate_topic, rounds=1) 
    manager.run_debate()