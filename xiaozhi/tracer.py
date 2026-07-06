"""调用路径记录器（call tracer）。

在 agent 运行期把每一次调用（用户轮次 / LLM / 工具 / MCP / 子agent /
后台任务 / cron 注入）组织成一棵树，实时写到 trace.json，供 HTML 页面轮询展示。

与原版差异：
- 目录不再硬编码到项目根，而是通过 ``tracer.configure(trace_dir=..., enabled=...)`` 注入；
  Agent 在初始化时按 config 调用一次。
- ``enabled=False`` 时所有对外 API 变为 no-op，零开销，便于库场景关闭。

设计要点：
- 线程安全：一把锁保护整棵树；用 threading.local 维护"每个线程的当前节点栈"。
- 零侵入兜底：写盘异常绝不影响 agent 主流程。
- 无第三方依赖：仅标准库。
"""

import os
import json
import time
import threading
from contextlib import contextmanager
from pathlib import Path

# 写盘节流：两次 flush 最短间隔（秒）。状态变更（结束）会强制 flush。
_FLUSH_MIN_INTERVAL = 0.15


def _now() -> float:
    return time.time()


class TraceNode:
    """调用树的一个节点。"""

    __slots__ = ("id", "parent_id", "type", "label", "detail",
                 "status", "start", "end", "children", "messages")

    def __init__(self, node_id, parent_id, node_type, label, detail=""):
        self.id = node_id
        self.parent_id = parent_id
        self.type = node_type          # turn|llm|tool|mcp|subagent|background|cron
        self.label = label
        self.detail = detail
        self.status = "running"        # running|ok|error
        self.start = _now()
        self.end = None
        self.children = []             # list[TraceNode]
        self.messages = None           # 该节点对应的 messages 快照（仅 llm 节点用）

    def to_dict(self):
        return {
            "id": self.id,
            "type": self.type,
            "label": self.label,
            "detail": self.detail,
            "status": self.status,
            "start": self.start,
            "end": self.end,
            "duration": (self.end - self.start) if self.end else None,
            "messages": self.messages,
            "children": [c.to_dict() for c in self.children],
        }


class Tracer:
    """全局调用树。单例，通过模块级 tracer 使用。

    默认 ``enabled=False``；由 Agent 在初始化时按配置 ``configure`` 打开。
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._counter = 0
        self._roots = []                       # list[TraceNode]
        self._local = threading.local()        # 每线程一个节点栈
        self._last_flush = 0.0
        self._started_at = _now()
        self._messages = []                    # 最近一次 messages 快照（供 HTML 展示）
        self._enabled = False
        self._trace_dir = None
        self._trace_file = None

    # ── 配置 ──

    def configure(self, trace_dir=None, enabled=True):
        """由 Agent 注入 trace 目录并开关记录。"""
        with self._lock:
            self._enabled = enabled
            if trace_dir is not None:
                self._trace_dir = Path(trace_dir)
                self._trace_file = self._trace_dir / "trace.json"

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ── 线程本地栈 ──

    def _stack(self):
        st = getattr(self._local, "stack", None)
        if st is None:
            st = []
            self._local.stack = st
        return st

    def _current(self):
        st = self._stack()
        return st[-1] if st else None

    def _next_id(self):
        self._counter += 1
        return f"n{self._counter}"

    def _add_node(self, node_type, label, detail, parent):
        """在锁内创建并挂载一个节点。返回 node。"""
        node = TraceNode(self._next_id(),
                         parent.id if parent else None,
                         node_type, label, detail)
        if parent:
            parent.children.append(node)
        else:
            self._roots.append(node)
        return node

    # ── 同步嵌套：上下文管理器 ──

    @contextmanager
    def span(self, node_type, label, detail=""):
        """记录一次同步调用。自动挂到当前线程栈顶节点下并压栈。"""
        if not self._enabled:
            yield None
            return
        with self._lock:
            parent = self._current()
            node = self._add_node(node_type, label, detail, parent)
            self._stack().append(node)
            self._flush_locked(force=True)
        try:
            yield node
        except Exception as e:
            with self._lock:
                node.status = "error"
                node.detail = (node.detail + f"\n{type(e).__name__}: {e}").strip()
                node.end = _now()
                self._pop(node)
                self._flush_locked(force=True)
            raise
        else:
            with self._lock:
                if node.status == "running":
                    node.status = "ok"
                node.end = _now()
                self._pop(node)
                self._flush_locked(force=True)

    def _pop(self, node):
        st = self._stack()
        if st and st[-1] is node:
            st.pop()
        elif node in st:
            st.remove(node)

    def set_detail(self, node, detail):
        """补充/更新某节点的 detail（如把 LLM 结果摘要写回）。"""
        if node is None:
            return
        with self._lock:
            node.detail = detail
            self._flush_locked()

    # ── 跨线程异步：后台任务 / cron worker ──

    def start_async(self, node_type, label, detail="", parent=None):
        """开启一个异步节点，挂到显式给定的 parent（或当前栈顶）下。"""
        if not self._enabled:
            return None
        with self._lock:
            if parent is None:
                parent = self._current()
            node = self._add_node(node_type, label, detail, parent)
            self._flush_locked(force=True)
            return node

    def seed_thread(self, node):
        """在新线程起点调用：把线程栈初始化为 [node]。"""
        self._local.stack = [node] if node else []

    def finish(self, node, status="ok", detail=None):
        """收尾一个异步节点。"""
        if node is None:
            return
        with self._lock:
            if detail is not None:
                node.detail = detail
            if node.status == "running":
                node.status = status
            node.end = _now()
            self._flush_locked(force=True)

    # ── messages / prompt 快照 ──

    @staticmethod
    def _serialize_messages(messages):
        """把 messages 精简成可 JSON 序列化的结构。"""
        snap = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            item = {"role": m.get("role", "")}
            content = m.get("content")
            item["content"] = content if isinstance(content, str) else (
                "" if content is None else str(content))
            if m.get("tool_calls"):
                item["tool_calls"] = [
                    {"name": tc.get("function", {}).get("name", ""),
                     "arguments": tc.get("function", {}).get("arguments", "")}
                    for tc in m["tool_calls"] if isinstance(tc, dict)
                ]
            if m.get("tool_call_id"):
                item["tool_call_id"] = m["tool_call_id"]
            snap.append(item)
        return snap

    def snapshot_messages(self, messages):
        """记录当前 messages（含 system prompt）的快照，供 HTML 面板展示。"""
        if not self._enabled:
            return
        try:
            snap = self._serialize_messages(messages)
        except Exception:
            return
        with self._lock:
            self._messages = snap
            self._flush_locked(force=True)

    def attach_messages(self, node, messages):
        """把某一轮实际发送给模型的 messages 快照挂到该节点（通常是 llm 节点）。"""
        if node is None:
            return
        try:
            snap = self._serialize_messages(messages)
        except Exception:
            return
        with self._lock:
            node.messages = snap
            self._flush_locked()

    # ── 写盘 ──

    def _flush_locked(self, force=False):
        if not self._enabled or self._trace_file is None:
            return
        now = _now()
        if not force and (now - self._last_flush) < _FLUSH_MIN_INTERVAL:
            return
        self._last_flush = now
        try:
            self._trace_dir.mkdir(parents=True, exist_ok=True)
            data = {
                "generated_at": now,
                "started_at": self._started_at,
                "roots": [r.to_dict() for r in self._roots],
                "messages": self._messages,
            }
            tmp = self._trace_file.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, self._trace_file)
        except Exception:
            # 写盘失败绝不影响主流程
            pass

    def reset(self):
        """清空当前树（一般无需调用；保留用于测试）。"""
        with self._lock:
            self._counter = 0
            self._roots = []
            self._local = threading.local()
            self._started_at = _now()
            self._messages = []
            self._flush_locked(force=True)


# 模块级单例
tracer = Tracer()