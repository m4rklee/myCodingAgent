"""工具调用执行器：解析参数 → 执行 → 追加 role:tool 消息。

支持后台派发（background_manager）与 hook（pre/post）。无全局依赖。
"""

from typing import Callable

from xiaozhi.tracer import tracer
from xiaozhi.tool_utils import parse_tool_arguments


def _trace_type(tool_name: str) -> str:
    """按工具名推断调用类型（用于树上的分类着色）。"""
    if tool_name.startswith("mcp__"):
        return "mcp"
    if tool_name == "spawn_subagent":
        return "subagent"
    return "tool"


def run_tool_calls(
    tool_calls,
    executor: Callable[[str, dict], str],
    messages: list,
    *,
    on_pre_hook=None,
    on_post_hook=None,
    background_manager=None,
    verbose: bool = False,
) -> list:
    """Parse arguments, execute tools, append role:tool messages.

    若提供 background_manager 且该工具满足后台条件，则派发到线程执行，
    tool 结果写占位符（保证每个 tool_call 都有配对的 role:tool 消息）。
    """
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
            elif background_manager is not None and \
                    background_manager.should_run_background(tool_name, arguments):
                # 后台任务：先在当前父节点下建一个异步节点，交给 background_manager 收尾
                trace_node = tracer.start_async(
                    "background", f"后台 · {tool_name}", detail=str(arguments)[:200])
                bg_id = background_manager.start(
                    tool_name, arguments, executor,
                    tool_use_id=tool_call["id"], trace_node=trace_node)
                tool_result = (f"[后台任务 {bg_id} 已启动] 工具：{tool_name}。"
                               f"结果将在完成后通过任务通知返回。")
            else:
                with tracer.span(_trace_type(tool_name), tool_name,
                                 detail=str(arguments)[:200]) as node:
                    tool_result = executor(tool_name, arguments)
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