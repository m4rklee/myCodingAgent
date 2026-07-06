"""OpenAI 兼容的异步流式 chat 调用。

与同步版 llm.py 对应，使用 AsyncOpenAI。model 显式传入。
tracer 复用同一单例（线程/协程安全由 tracer 内部锁保证）。
"""

from xiaozhi.tracer import tracer


async def achat_completion_stream(
    client,
    messages,
    tools=None,
    model=None,
    statistics=None,
    print_output=True,
    **kwargs,
):
    """发送异步流式请求，收集文本 / 工具调用 / finish_reason。"""
    with tracer.span("llm", f"LLM · {model}") as node:
        tracer.attach_messages(node, messages)
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            tools=tools,
            stream_options={"include_usage": True},
            **kwargs,
        )
        llm_response, tool_calls, finish_reason = await aparse_chat_completion_stream(
            response, statistics=statistics, print_output=print_output)
        summary = []
        if tool_calls:
            names = ", ".join(tc["function"]["name"] for tc in tool_calls)
            summary.append(f"→ 请求工具: {names}")
        if llm_response:
            text = llm_response.strip().replace("\n", " ")
            summary.append(f"文本: {text[:160]}")
        summary.append(f"finish_reason={finish_reason}")
        tracer.set_detail(node, "\n".join(summary))
        return llm_response, tool_calls, finish_reason


async def aparse_chat_completion_stream(response, statistics=None, print_output=True):
    """解析异步流式响应。response 为 async iterator。"""
    llm_response = ""
    tool_calls = {}
    finish_reason = None

    async for chunk in response:
        if getattr(chunk, "usage", None) and statistics:
            statistics.update_token_usage(chunk.usage)

        if not chunk.choices:
            continue

        choice = chunk.choices[0]
        delta = choice.delta

        if choice.finish_reason:
            finish_reason = choice.finish_reason

        if delta.content:
            llm_response += delta.content
            if print_output:
                print(delta.content, end='', flush=True)

        if delta.tool_calls:
            for tool_call_delta in delta.tool_calls:
                index = tool_call_delta.index
                if index not in tool_calls:
                    tool_calls[index] = {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                if tool_call_delta.id:
                    tool_calls[index]["id"] += tool_call_delta.id
                if tool_call_delta.type:
                    tool_calls[index]["type"] = tool_call_delta.type
                if tool_call_delta.function:
                    if tool_call_delta.function.name:
                        tool_calls[index]["function"]["name"] += tool_call_delta.function.name
                    if tool_call_delta.function.arguments:
                        tool_calls[index]["function"]["arguments"] += tool_call_delta.function.arguments

    if print_output:
        print()

    return llm_response, list(tool_calls.values()), finish_reason