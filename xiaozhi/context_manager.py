"""多层上下文压缩管理器。

与原版差异：``transcript_dir`` 与 LLM ``client``/``model`` 通过构造函数注入，
不再依赖全局 config。
"""

import json
import time
from pathlib import Path

KEEP_RECENT_TOOL_RESULTS = 3


class AgentContextManager:
    def __init__(self, client, model, transcript_dir: Path,
                 total: int = 1_000_000, threshold: int = 10_000):
        self.client = client
        self.model = model
        self.transcript_dir = Path(transcript_dir)
        self.TOTAL = total          # 上下文窗口上限（估算）
        self.THRESHOLD = threshold  # 超过此阈值触发压缩，留出缓冲
        self.cur_window: int = 0

    def display_context_window(self):
        print(f'上下文窗口占用量：{self.cur_window} / 1M')

    def update_context(self, tokens):
        self.cur_window += tokens

    def clear_context(self):
        self.cur_window = 0

    def _message_has_tool_use(self, message):
        if message.get('role') != "assistant":
            return False
        return bool(message.get("tool_calls"))

    def _is_tool_result_message(self, message):
        return message.get('role') == 'tool'

    def collect_tool_result_response(self, messages):
        tool_result_messages = []
        for message in messages:
            if message.get('role') == 'tool':
                tool_result_messages.append(message)
        return tool_result_messages

    # 第一类压缩
    # 1.裁掉中间记录，但要保证tool_calls和tool消息成对
    def snip_compact(self, messages, max_messages=50):
        if len(messages) <= max_messages:
            return messages

        head_end, tail_start = 1, len(messages) - (max_messages - 1)

        # head 这边：如果保留区最后一条是 assistant(tool_calls)，
        # 就把它后面的 tool 结果也一起保留，避免 assistant 声明了 tool_calls 但结果被裁掉。
        if head_end > 0 and self._message_has_tool_use(messages[head_end - 1]):
            while head_end < len(messages) and self._is_tool_result_message(messages[head_end]):
                head_end += 1

        # tail 这边：如果 tail_start 落在 OpenAI 的连续 tool 消息中间，
        # 就一直往前退，直到退到 assistant(tool_calls)。
        if tail_start > 0 and tail_start < len(messages):
            while tail_start > 0 and self._is_tool_result_message(messages[tail_start]):
                tail_start -= 1

        snipped = tail_start - head_end
        placeholder = {
            "role": "user",
            "content": f"[裁剪掉会话中间的 {snipped} 条消息]"
        }

        return messages[:head_end] + [placeholder] + messages[tail_start:]

    # 第二类压缩
    def micro_compact(self, messages):
        tool_results = self.collect_tool_result_response(messages)
        if len(tool_results) <= KEEP_RECENT_TOOL_RESULTS:
            return messages
        for msg in tool_results[:-KEEP_RECENT_TOOL_RESULTS]:
            if len(str(msg.get("content", ""))) > 120:
                msg["content"] = "工具调用结果已压缩，必要时需要重新执行"
        return messages

    # 第三类压缩
    def tool_result_budget(self, messages, max_bytes=200_000):
        # OpenAI: 工具结果是结尾连续的多条 role="tool" 消息
        blocks = []
        for i in range(len(messages) - 1, -1, -1):
            if self._is_tool_result_message(messages[i]):
                blocks.append(messages[i])
            else:
                break  # 遇到非 tool 消息（assistant tool_calls 等）就停
        blocks.reverse()

        if not blocks:
            return messages

        def size_of(msg):
            return len(str(msg.get("content", "")))

        total = sum(size_of(m) for m in blocks)
        if total <= max_bytes:
            return messages

        # 从最大的开始落盘
        ranked = sorted(blocks, key=size_of, reverse=True)
        for msg in ranked:
            if total <= max_bytes:
                break
            msg["content"] = self.persist_large_output(
                msg.get("tool_call_id", "unknown"),
                str(msg.get("content", "")),
            )
            total = sum(size_of(m) for m in blocks)

        return messages

    def persist_large_output(self, tool_call_id, content, preview_len=2000):
        persist_dir = self.transcript_dir.parent / ".task_outputs" / "tool-results"
        persist_dir.mkdir(parents=True, exist_ok=True)
        path = persist_dir / f"{tool_call_id}.txt"
        path.write_text(content, encoding="utf-8")

        preview = content[:preview_len]
        return (
            f'<persisted-output tool_call_id="{tool_call_id}" '
            f'path="{path}" total_chars="{len(content)}">\n'
            f"{preview}\n"
            f"...(完整内容已落盘，需要时可用 read_file 读取 {path})\n"
            f"</persisted-output>"
        )

    # 第四类压缩
    def write_transcript(self, messages):
        self.transcript_dir.mkdir(parents=True, exist_ok=True)
        path = self.transcript_dir / f"transcript_{int(time.time())}.jsonl"
        with path.open("w") as f:
            for msg in messages:
                f.write(json.dumps(msg, default=str) + "\n")
        return path

    def summarize_history(self, messages):
        conversation = json.dumps(messages, default=str)[:80000]
        prompt = ("在保证工作能继续的前提下总结当前的对话.\n"
                  "保留以下: 1. 当前目标, 2. 关键发现/决定, 3. 读或改变的文件, "
                  "4. 剩余工作, 5. 用户约束.\n压缩的同时保持具体.\n\n" + conversation)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
        )
        return (response.choices[0].message.content or "").strip() or "(empty summary)"

    def compact_history(self, messages):
        self.write_transcript(messages)
        summary = self.summarize_history(messages)

        new_messages = []
        # 保留 system prompt（人设 + 工具说明），避免摘要后丢失
        if messages and messages[0].get("role") == "system":
            new_messages.append(messages[0])
        new_messages.append({
            "role": "user",
            "content": f"[压缩结果]\n\n{summary}"
        })
        return new_messages

    def compact_context(self, messages):
        """
        多层上下文压缩机制
        L3.大结果落盘  -> tool_result_budget
        L1.裁中间      -> snip_compact
        L2.旧结果占位  -> micro_compact
        L4.LLM全量摘要 -> compact_history
        """
        # 三个预处理器（0 API 调用）
        # 顺序：budget 先跑，确保大内容落盘后再做裁剪和占位
        messages[:] = self.tool_result_budget(messages)   # L3: 大结果落盘
        messages[:] = self.snip_compact(messages)         # L1: 裁中间
        messages[:] = self.micro_compact(messages)        # L2: 旧结果占位

        # 还不够？LLM 摘要（1 API 调用）
        if self.calculate_messages_tokens(messages) > self.THRESHOLD:
            messages[:] = self.compact_history(messages)

        return messages

    def calculate_tokens(self, text):
        if text is None:
            return 0
        return len(str(text)) // 4

    def calculate_message_tokens(self, message):
        """估算单条 message 的 token 数。"""
        role_tokens = self.calculate_tokens(message.get("role", ""))
        content_tokens = self.calculate_tokens(message.get("content", ""))

        tool_calls = message.get("tool_calls")
        tool_call_tokens = self.calculate_tokens(json.dumps(tool_calls, default=str)) if tool_calls else 0

        # role/content 等 ChatML 包装格式会有额外开销，这里先粗略加 4
        return role_tokens + content_tokens + tool_call_tokens + 4

    def calculate_messages_tokens(self, messages):
        """估算整个 messages 当前占用的 token 数。"""
        return sum(self.calculate_message_tokens(message) for message in messages)

    def refresh_context_window(self, messages):
        """重新计算当前上下文窗口占用量。"""
        self.cur_window = self.calculate_messages_tokens(messages)
        return self.cur_window

    def ensure_context_limit(self, messages):
        """保证 messages 不超过上下文窗口限制。超过 THRESHOLD 就触发多层压缩。"""
        self.refresh_context_window(messages)

        if self.cur_window > self.THRESHOLD:
            messages = self.compact_context(messages)
            self.refresh_context_window(messages)

        return messages

    # ── 异步压缩路径 ──
    # AsyncAgent 注入的是 AsyncOpenAI，同步的 summarize_history 会崩
    #（`create()` 返回 coroutine，访问 .choices 直接 AttributeError）。
    # 以下 async 变体用 await 调用，供 AsyncAgent 使用。

    async def asummarize_history(self, messages):
        conversation = json.dumps(messages, default=str)[:80000]
        prompt = ("在保证工作能继续的前提下总结当前的对话.\n"
                  "保留以下: 1. 当前目标, 2. 关键发现/决定, 3. 读或改变的文件, "
                  "4. 剩余工作, 5. 用户约束.\n压缩的同时保持具体.\n\n" + conversation)
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
        )
        return (response.choices[0].message.content or "").strip() or "(empty summary)"

    async def acompact_history(self, messages):
        self.write_transcript(messages)
        summary = await self.asummarize_history(messages)
        new_messages = []
        if messages and messages[0].get("role") == "system":
            new_messages.append(messages[0])
        new_messages.append({"role": "user", "content": f"[压缩结果]\n\n{summary}"})
        return new_messages

    async def acompact_context(self, messages):
        """多层压缩的异步版：前三层无 API 调用，最后一层用 await 摘要。"""
        messages[:] = self.tool_result_budget(messages)
        messages[:] = self.snip_compact(messages)
        messages[:] = self.micro_compact(messages)
        if self.calculate_messages_tokens(messages) > self.THRESHOLD:
            messages[:] = await self.acompact_history(messages)
        return messages

    async def aensure_context_limit(self, messages):
        """ensure_context_limit 的异步版；AsyncAgent 应调用它。"""
        self.refresh_context_window(messages)
        if self.cur_window > self.THRESHOLD:
            messages = await self.acompact_context(messages)
            self.refresh_context_window(messages)
        return messages

    def append_message(self, messages, role=None, content=None, *, message=None):
        """仅追加消息并累加 token 计数。压缩不在这里做——
        它只应在「向 LLM 发请求前」的安全边界统一触发（见 loop）。

        两种用法：
        - append_message(messages, role, content)
        - append_message(messages, message={...})
        """
        if message is None:
            message = {"role": role, "content": content}
        messages.append(message)
        self.update_context(self.calculate_message_tokens(message))
        return messages