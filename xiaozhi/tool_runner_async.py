"""异步工具调用执行器：解析参数 → await 执行 → 追加 role:tool 消息。

与同步版 tool_runner.py 对应。executor 是一个 async callable (name, arguments) -> str。
不支持后台派发（异步场景用 asyncio.create_task 更自然，编排层负责）。
"""

from typing import Awaitable, Callable

from xiaozhi.tracer import tracer
from xiaozhi.tool_utils import parse_tool_arguments


def _trace_type(tool_name: str) -> str:
    if tool_name.startswith("mcp__"):
        return "mcp"
    if tool_name == "spawn_subagent":
        return "subagent"
    return "tool"


async def arun_tool_calls(
    tool_calls,
    executor: Callable[[str, dict], Awaitable[str]],
    messages: list,
    *,
    on_pre_hook=None,
    on_post_hook=None,
    verbose: bool = False,
) -> list:
    """解析参数、await 执行工具、追加 role:tool 消息。"""
    for tool_call in tool_calls:
        tool_name = tool_call["function"]["name"]
        raw_arguments = tool_call["function"]["arguments"]

        arguments, parse_error = parse_tool_arguments(raw_arguments)
        if parse_error:
            tool_result = parse_error
        else:
            if verbose:
                print(f"模型调用工具：{tool_name}")
                print(f"工具参数：{arguments}")
            denial = on_pre_hook(tool_call) if on_pre_hook else None
            if denial:
                tool_result = denial
            else:
                with tracer.span(_trace_type(tool_name), tool_name,
                                 detail=str(arguments)[:200]) as node:
                    tool_result = await executor(tool_name, arguments)
                    tracer.set_detail(
                        node, f"{str(arguments)[:200]}\n→ {str(tool_result)[:300]}")
            if on_post_hook:
                on_post_hook(tool_call, tool_result)

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": tool_result,
        })

    return messages