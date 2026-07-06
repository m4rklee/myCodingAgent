"""Hook 管理：UserPromptSubmit / PreToolUse / PostToolUse / Stop 四类事件。

与原版差异：改为 ``HookManager`` 类，不再 import 时全局注册。
Agent 持有一个实例；默认注册权限校验、日志、大输出告警等内置 hook，
也可通过 ``register`` 追加自定义 hook。
"""

from pathlib import Path

from xiaozhi.tool_utils import parse_openai_tool_call

DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if="]
DESTRUCTIVE = ["rm", "> /etc/", "chmod 777"]


def _confirm_action() -> bool:
    choice = input("   Allow? [y/N] ").strip().lower()
    return choice in ("y", "yes")


class HookManager:
    """按事件分发 hook；任一 hook 返回非 None 即短路（用于拒绝工具调用）。"""

    def __init__(self, workdir: Path, install_defaults: bool = True,
                 confirm_fn=None):
        self.workdir = Path(workdir)
        self.confirm_fn = confirm_fn or _confirm_action
        self.hooks = {"UserPromptSubmit": [], "PreToolUse": [],
                      "PostToolUse": [], "Stop": []}
        if install_defaults:
            self.register("PreToolUse", self.permission_hook)
            self.register("PreToolUse", self.log_hook)
            self.register("PostToolUse", self.large_output_hook)
            self.register("UserPromptSubmit", self.context_inject_hook)
            self.register("Stop", self.summary_hook)

    def register(self, event: str, callback):
        if event not in self.hooks:
            raise ValueError(f"未知 hook 事件：{event}")
        self.hooks[event].append(callback)

    def trigger(self, event: str, *args):
        for callback in self.hooks[event]:
            result = callback(*args)
            if result is not None:
                return result
        return None

    # ── 内置 hook ──

    def permission_hook(self, tool_call):
        """PreToolUse: check permission for OpenAI-compatible tool calls."""
        tool_name, arguments = parse_openai_tool_call(tool_call)
        if tool_name == "bash":
            command = arguments.get("command", "")
            for pattern in DENY_LIST:
                if pattern in command:
                    print(f"\n\033[31m⛔ Blocked: '{pattern}'\033[0m")
                    return "Permission denied by deny list"
            for kw in DESTRUCTIVE:
                if kw in command:
                    print(f"\n\033[33m⚠  Potentially destructive command\033[0m")
                    print(f"   Tool: {tool_name}({arguments})")
                    if not self.confirm_fn():
                        return "Permission denied by user"
        if tool_name in ("write_file", "edit_file"):
            path = arguments.get("path", "")
            if not (self.workdir / path).resolve().is_relative_to(self.workdir):
                print(f"\n\033[33m⚠  Writing outside workspace\033[0m")
                print(f"   Tool: {tool_name}({arguments})")
                if not self.confirm_fn():
                    return "Permission denied by user"
        return None

    def log_hook(self, tool_call):
        """PreToolUse: log every OpenAI-compatible tool call."""
        tool_name, arguments = parse_openai_tool_call(tool_call)
        args_preview = str(list(arguments.values())[:2])[:60]
        print(f"\033[90m[HOOK] {tool_name}({args_preview})\033[0m")
        return None

    def large_output_hook(self, tool_call, output):
        """PostToolUse: warn on large output."""
        tool_name, _ = parse_openai_tool_call(tool_call)
        if len(str(output)) > 100000:
            print(f"\033[33m[HOOK] ⚠ Large output from {tool_name}: {len(str(output))} chars\033[0m")
        return None

    def context_inject_hook(self, query: str):
        print(f"\033[90m[HOOK] UserPromptSubmit: working in {self.workdir}\033[0m")
        return None

    def summary_hook(self, messages: list):
        tool_count = sum(1 for m in messages if m.get("role") == "tool")
        print(f"\033[90m[HOOK] Stop: session used {tool_count} tool calls\033[0m")
        return None