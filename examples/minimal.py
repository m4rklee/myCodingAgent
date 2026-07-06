"""最小示例：一个没有自定义工具的 Agent。"""

from xiaozhi import Agent

# 默认从环境变量 LLM_MODEL / LLM_API_KEY / LLM_BASE_URL 读取配置
agent = Agent()

reply = agent.chat("你好，请用一句话介绍你自己")
print(f"\n回答：{reply}")