"""命令行入口：`xiaozhi` 启动一个交互式 REPL。"""

import argparse

from xiaozhi.agent import Agent
from xiaozhi.config import AgentConfig


def main():
    parser = argparse.ArgumentParser(description="小智 Agent 交互式命令行")
    parser.add_argument("--model", default=None, help="模型名（默认读 LLM_MODEL 环境变量）")
    parser.add_argument("--cron", action="store_true", help="启用 cron 定时任务")
    parser.add_argument("--worktree", action="store_true", help="启用 git worktree")
    parser.add_argument("--mcp", action="store_true", help="启用 MCP 工具")
    parser.add_argument("--trace", action="store_true", help="启用调用 trace 记录")
    parser.add_argument("--trace-server", action="store_true", help="启动 trace 可视化网页")
    args = parser.parse_args()

    config = AgentConfig(
        model=args.model,
        enable_cron=args.cron,
        enable_worktree=args.worktree,
        enable_mcp=args.mcp,
        enable_trace=args.trace or args.trace_server,
        enable_trace_server=args.trace_server,
    )
    Agent(config=config).repl()


if __name__ == "__main__":
    main()