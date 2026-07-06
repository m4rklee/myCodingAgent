"""完整功能示例：开启所有可选模块。"""

from xiaozhi import Agent, AgentConfig

config = AgentConfig(
    model="gpt-4o",
    enable_cron=True,
    enable_worktree=True,
    enable_mcp=True,
    enable_trace=True,
    enable_trace_server=True,
    enable_memory=True,
)

agent = Agent(config=config)

print(agent.chat("帮我列出所有可用的 MCP server"))
# agent.repl()  # 交互模式