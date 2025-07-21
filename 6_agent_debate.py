'''
Descripttion: 说明
version: V1.0
Author: StarryLei
Date: 2025-07-22 00:06:16
LastEditors: StarryLei
LastEditTime: 2025-07-22 00:06:43
'''
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import json
import time
import os
from dotenv import load_dotenv

load_dotenv(override=True)

if not os.environ.get("ARK_API_KEY"):
    raise ValueError("请设置 ARK_API_KEY 环境变量")

ARK_API_KEY = os.getenv("ARK_API_KEY")

# --- 裁判的提示 ---
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


# --- 辩手的提示 ---
debater_template = """
你是一位顶级的辩论选手。你的风格是逻辑严密，言辞犀利，同时富有说服力。

# 辩论总议题: {topic}
# 你的立场: {stance}

# 你的任务:
- **陈述观点**: 清晰地阐述支持你立场的论点。
- **提供论据**: 使用事实、数据、逻辑推理或生动的例子来支撑你的论点。
- **反驳对手**: 如果辩论历史中有对手的观点，你需要有针对性地进行反驳。
- **保持角色**: 始终坚定地捍卫你的立场，不要动摇。

# 辩论历史:
{history}

---
现在轮到你发言了。请开始你的陈述，直接输出你的辩词。
"""
debater_prompt = ChatPromptTemplate.from_template(debater_template)

# 为了让辩论更有趣，可以为不同角色使用不同模型或温度设置
# 但为简单起见，我们这里使用同一个配置
llm = ChatOpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=ARK_API_KEY,
        model="ep-m-20250719172710-9zfxx",
        streaming=True
    )

# 创建三个核心的 Agent Chain
referee_agent = referee_prompt | llm
pro_debater_agent = debater_prompt | llm
con_debater_agent = debater_prompt | llm


class DebateManager:
    def __init__(self, topic, rounds=4):
        self.topic = topic
        self.rounds = rounds
        self.history = ""
        self.scores = {"pro": 0, "con": 0}
        self.pro_stance = ""
        self.con_stance = ""

    def _call_referee(self, task):
        """调用裁判并解析其JSON输出"""
        response = referee_agent.invoke({
            "topic": self.topic,
            "history": self.history,
            "task": task
        })
        try:
            # 去掉可能的代码块标记
            cleaned_response = response.content.strip().replace("```json", "").replace("```", "")
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            print(f"裁判返回了非JSON格式的回应: {response.content}")
            return None

    def setup_debate(self):
        """让裁判生成正反方立场"""
        print("--- 辩论准备阶段 ---")
        print(f"裁判正在根据议题 '{self.topic}' 生成双方立场...")
        
        task = "请为本次辩论生成正反双方的立场。"
        stances = self._call_referee(task)
        
        if stances and "pro_stance" in stances and "con_stance" in stances:
            self.pro_stance = stances["pro_stance"]
            self.con_stance = stances["con_stance"]
            self.history += f"总议题: {self.topic}\n"
            self.history += f"正方立场: {self.pro_stance}\n"
            self.history += f"反方立场: {self.con_stance}\n\n"
            print(f"正方立场: {self.pro_stance}")
            print(f"反方立场: {self.con_stance}")
        else:
            raise ValueError("无法从裁判处获得有效的辩题。")

    def run_round(self, round_num):
        """运行一轮辩论"""
        print(f"\n--- 第 {round_num} 轮辩论开始 ---")

        # 正方发言
        pro_argument = pro_debater_agent.invoke({
            "topic": self.topic,
            "stance": self.pro_stance,
            "history": self.history
        }).content
        print(f"\n[正方辩手]:\n{pro_argument}")
        self.history += f"第{round_num}轮 - 正方: {pro_argument}\n"
        time.sleep(1) # 模拟思考

        # 裁判为正方评分
        pro_score_task = f"请为刚才正方辩手的陈述评分。"
        pro_score_result = self._call_referee(pro_score_task)
        if pro_score_result:
            score = pro_score_result.get('score', 0)
            reasoning = pro_score_result.get('reasoning', '无理由')
            self.scores["pro"] += score
            print(f"\n[裁判点评-正方]:\n分数: {score}/15\n理由: {reasoning}\n")
            self.history += f"裁判评分(正方): {score}/15, 理由: {reasoning}\n\n"
        
        time.sleep(15) # 等待一下，避免API速率限制，也让流程更清晰

        # 反方发言
        con_argument = con_debater_agent.invoke({
            "topic": self.topic,
            "stance": self.con_stance,
            "history": self.history
        }).content
        print(f"\n[反方辩手]:\n{con_argument}")
        self.history += f"第{round_num}轮 - 反方: {con_argument}\n"
        time.sleep(1)

        # 裁判为反方评分
        con_score_task = f"请为刚才反方辩手的陈述评分。"
        con_score_result = self._call_referee(con_score_task)
        if con_score_result:
            score = con_score_result.get('score', 0)
            reasoning = con_score_result.get('reasoning', '无理由')
            self.scores["con"] += score
            print(f"\n[裁判点评-反方]:\n分数: {score}/15\n理由: {reasoning}\n")
            self.history += f"裁判评分(反方): {score}/15, 理由: {reasoning}\n\n"
        
        time.sleep(15)

    def announce_winner(self):
        """宣布最终结果"""
        print("\n--- 辩论结束 ---")
        print(f"最终得分 -> 正方: {self.scores['pro']} | 反方: {self.scores['con']}")
        
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
        
        print("\n[裁判最终裁决]:")
        print(summary_response.content)

    def run_debate(self):
        """完整运行整个辩论"""
        self.setup_debate()
        for i in range(1, self.rounds + 1):
            self.run_round(i)
        self.announce_winner()


if __name__ == "__main__":
    # 用户输入一个辩论议题
    debate_topic = "人工智能的发展对人类社会是利大于弊还是弊大于利"
    
    manager = DebateManager(topic=debate_topic, rounds=4)
    manager.run_debate()