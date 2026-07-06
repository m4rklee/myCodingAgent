"""后台任务：线程执行 + 通知注入（OpenAI 兼容格式）。

以 (tool_name, arguments) 为单位派发，工具执行走 executor(name, arguments)。
无全局依赖。
"""

import threading
from typing import Any, Callable

from xiaozhi.tracer import tracer

# 慢操作启发式：bash 里出现下列关键词，或本身就耗时/输出多的工具
SLOW_BASH_KEYWORDS = ["install", "build", "test", "deploy", "compile",
                      "docker build", "pip install", "npm install",
                      "cargo build", "pytest", "make"]
SLOW_TOOLS = {"spawn_subagent"}


def is_slow_operation(tool_name: str, tool_input: dict) -> bool:
    """兜底启发式：预计耗时 > 30s 的操作。"""
    if tool_name in SLOW_TOOLS:
        return True
    if tool_name != "bash":
        return False
    cmd = str(tool_input.get("command", "")).lower()
    return any(kw in cmd for kw in SLOW_BASH_KEYWORDS)


def should_run_background(tool_name: str, tool_input: dict) -> bool:
    """模型显式请求优先；否则回退到启发式。"""
    if tool_input.get("run_in_background"):
        return True
    return is_slow_operation(tool_name, tool_input)


class BackgroundTaskManager:
    """线程执行后台任务，并把完成结果收集成通知消息。"""

    def __init__(self):
        self._counter = 0
        self._lock = threading.Lock()
        self.tasks: dict[str, dict] = {}      # bg_id → {tool_use_id, command, status}
        self.results: dict[str, str] = {}     # bg_id → output

    @staticmethod
    def _command_of(tool_name: str, arguments: dict) -> str:
        """给后台任务取一个可读的标签。"""
        return str(arguments.get("command")
                   or arguments.get("task_description")
                   or tool_name)

    def should_run_background(self, tool_name: str, arguments: dict) -> bool:
        return should_run_background(tool_name, arguments)

    def start(self, tool_name: str, arguments: dict, executor: Callable[[str, dict], str],
              tool_use_id: str = "", trace_node=None) -> str:
        """在守护线程中执行工具，返回后台任务 ID。"""
        with self._lock:
            self._counter += 1
            bg_id = f"bg_{self._counter:04d}"
            self.tasks[bg_id] = {
                "tool_use_id": tool_use_id,
                "command": self._command_of(tool_name, arguments),
                "status": "running",
            }

        def worker():
            tracer.seed_thread(trace_node)
            status = "ok"
            try:
                result = executor(tool_name, arguments)
            except Exception as e:
                status = "error"
                result = f"Error: {type(e).__name__}: {e}"
            tracer.finish(trace_node, status,
                          detail=f"{status} · {len(result)} chars\n{result[:300]}")
            with self._lock:
                self.tasks[bg_id]["status"] = "completed"
                self.results[bg_id] = result

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        cmd = self.tasks[bg_id]["command"]
        print(f"  \033[33m[background] 已派发 {bg_id}: {cmd[:40]}\033[0m")
        return bg_id

    def collect_notifications(self) -> list[str]:
        """收集已完成的后台任务，返回 <task_notification> 文本列表。"""
        with self._lock:
            ready_ids = [bid for bid, task in self.tasks.items()
                         if task["status"] == "completed"]

        notifications = []
        for bg_id in ready_ids:
            with self._lock:
                task = self.tasks.pop(bg_id)
                output = self.results.pop(bg_id, "")
            summary = output[:200] if len(output) > 200 else output
            notifications.append(
                f"<task_notification>\n"
                f"  <task_id>{bg_id}</task_id>\n"
                f"  <status>completed</status>\n"
                f"  <command>{task['command']}</command>\n"
                f"  <summary>{summary}</summary>\n"
                f"</task_notification>"
            )
            print(f"  \033[32m[background done] {bg_id}: "
                  f"{task['command'][:40]} ({len(output)} chars)\033[0m")
        return notifications