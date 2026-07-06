"""OpenAI 兼容的流式 chat 调用与解析。

与原版差异：``model`` 显式传入，不再有 ``os.getenv`` 兜底——
模型来源统一收敛到 AgentConfig。
"""

from xiaozhi.tracer import tracer


def chat_completion_stream(
    client,
    messages,
    tools=None,
    model=None,
    statistics=None,
    print_output=True,
    **kwargs,
):
    """Send an OpenAI-compatible streaming chat request and collect text/tool calls."""
    with tracer.span("llm", f"LLM · {model}") as node:
        # 记录本轮实际发送给模型的 prompt（messages）
        tracer.attach_messages(node, messages)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            tools=tools,
            stream_options={"include_usage": True},
            **kwargs,
        )
        llm_response, tool_calls, finish_reason = parse_chat_completion_stream(
            response, statistics=statistics, print_output=print_output)
        # 把结果摘要写回节点 detail
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


def parse_chat_completion_stream(response, statistics=None, print_output=True):
    """Parse an OpenAI-compatible streaming response into text, tool calls and finish_reason."""
    llm_response = ""
    tool_calls = {}
    finish_reason = None

    for chunk in response:
        if getattr(chunk, "usage", None) and statistics:
            statistics.update_token_usage(chunk.usage)

        if not chunk.choices:
            continue

        choice = chunk.choices[0]
        delta = choice.delta

        # 结束原因（通常最后一个 chunk 才非空）；"length" 表示触达 max_tokens
        if choice.finish_reason:
            finish_reason = choice.finish_reason

        # 最终回答
        if delta.content:
            llm_response += delta.content
            if print_output:
                print(delta.content, end='')

        if delta.tool_calls:
            for tool_call_delta in delta.tool_calls:
                index = tool_call_delta.index

                if index not in tool_calls:
                    tool_calls[index] = {
                        "id": "",
                        "type": "function",
                        "function": {
                            "name": "",
                            "arguments": "",
                        }
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