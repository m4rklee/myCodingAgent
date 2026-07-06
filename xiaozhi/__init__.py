"""小智 —— 一个自研的、可嵌入的轻量 Agent 框架（OpenAI 兼容）。

快速开始：
    from xiaozhi import Agent, tool

    @tool(description="查询城市天气")
    def get_weather(city: str) -> str:
        return f"{city} 今天晴，26℃"

    agent = Agent(model="gpt-4o", tools=[get_weather])
    print(agent.chat("北京天气怎么样？"))

异步 + 多 Agent 编排：
    from xiaozhi import AsyncAgent, AgentTeam
"""

from xiaozhi.agent import Agent
from xiaozhi.aio import AsyncAgent
from xiaozhi.config import AgentConfig
from xiaozhi.decorators import tool, ToolSpec
from xiaozhi.orchestration import AgentTeam, Subtask, WorkerResult, TeamResult

__version__ = "0.2.0"
__all__ = [
    "Agent", "AsyncAgent", "AgentConfig", "tool", "ToolSpec",
    "AgentTeam", "Subtask", "WorkerResult", "TeamResult",
]