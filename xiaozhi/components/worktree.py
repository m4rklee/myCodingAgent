"""git worktree 隔离：为任务创建独立分支目录，避免多 Agent 并发改同一文件冲突。

与原版差异：``workdir`` 与 ``worktrees_dir`` 通过构造函数注入。
生命周期事件写入 .worktrees/events.jsonl。
"""

import json
import re
import subprocess
import time
from pathlib import Path

from xiaozhi.tool_utils import require_param
from xiaozhi.components.tasks import AgentTask

# 合法 worktree 名：字母、数字、点、下划线、短横，1-64 字符
VALID_WT_NAME = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class AgentWorktree:
    """git worktree 隔离：为任务创建独立分支目录。"""

    def __init__(self, workdir: Path, worktrees_dir: Path, tasks: AgentTask):
        self.workdir = Path(workdir)
        self.worktrees_dir = Path(worktrees_dir)
        self.worktrees_dir.mkdir(parents=True, exist_ok=True)
        self.tasks = tasks

    # ── 校验 & git 封装 ──

    def validate_worktree_name(self, name: str) -> str | None:
        """校验 worktree 名，拒绝路径穿越和非法字符。返回错误信息或 None。"""
        if not name:
            return "worktree 名不能为空"
        if name in (".", ".."):
            return f"'{name}' 不是合法的 worktree 名"
        if not VALID_WT_NAME.match(name):
            return (f"非法 worktree 名 '{name}'："
                    "只允许字母、数字、点、下划线、短横（1-64 字符）")
        return None

    def run_git(self, args: list[str], cwd: Path | None = None) -> tuple[bool, str]:
        """执行 git 命令，返回 (ok, output)。"""
        try:
            r = subprocess.run(["git"] + args, cwd=cwd or self.workdir,
                               capture_output=True, text=True, timeout=30)
            out = (r.stdout + r.stderr).strip()
            out = out[:5000] if out else "(no output)"
            return r.returncode == 0, out
        except subprocess.TimeoutExpired:
            return False, "Error: git 超时"

    def log_event(self, event_type: str, worktree_name: str, task_id: str = ""):
        """把生命周期事件追加到 events.jsonl。"""
        event = {"type": event_type, "worktree": worktree_name,
                 "task_id": task_id, "ts": time.time()}
        events_file = self.worktrees_dir / "events.jsonl"
        with open(events_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    # ── 核心操作 ──

    def create_worktree(self, name: str, task_id: str = "") -> str:
        """创建带独立分支的 git worktree，可选绑定到任务。"""
        err = self.validate_worktree_name(name)
        if err:
            return f"Error: {err}"
        path = self.worktrees_dir / name
        if path.exists():
            return f"Worktree '{name}' 已存在于 {path}"
        ok, result = self.run_git(
            ["worktree", "add", str(path), "-b", f"wt/{name}", "HEAD"])
        if not ok:
            return f"Git error: {result}"
        if task_id:
            self.bind_task_to_worktree(task_id, name)
        self.log_event("create", name, task_id)
        print(f"  \033[33m[worktree] created: {name} at {path}\033[0m")
        return f"Worktree '{name}' created at {path}"

    def bind_task_to_worktree(self, task_id: str, worktree_name: str):
        """把 worktree 字段写入任务，保持 pending 状态以便自动认领。"""
        task = self.tasks.load_task(task_id)
        task.worktree = worktree_name
        self.tasks.save_task(task)
        print(f"  \033[33m[bind] {task.subject} → worktree:{worktree_name}\033[0m")

    def _count_worktree_changes(self, path: Path) -> tuple[int, int]:
        """统计 worktree 中未提交文件数与未推送提交数。出错返回 (-1, -1)。"""
        try:
            r1 = subprocess.run(["git", "status", "--porcelain"],
                                cwd=path, capture_output=True, text=True, timeout=10)
            files = len([l for l in r1.stdout.strip().splitlines() if l.strip()])
            r2 = subprocess.run(["git", "log", "@{push}..HEAD", "--oneline"],
                                cwd=path, capture_output=True, text=True, timeout=10)
            commits = len([l for l in r2.stdout.strip().splitlines() if l.strip()])
            return files, commits
        except Exception:
            return -1, -1

    def remove_worktree(self, name: str, discard_changes: bool = False) -> str:
        """删除 worktree。若有未提交改动则拒绝，除非 discard_changes=True。"""
        err = self.validate_worktree_name(name)
        if err:
            return err
        path = self.worktrees_dir / name
        if not path.exists():
            return f"Worktree '{name}' 不存在"
        if not discard_changes:
            files, commits = self._count_worktree_changes(path)
            if files < 0:
                return (f"无法确认 worktree '{name}' 状态。"
                        "使用 discard_changes=true 强制删除。")
            if files > 0 or commits > 0:
                return (f"Worktree '{name}' 有 {files} 个未提交文件、"
                        f"{commits} 个未推送提交。"
                        "使用 discard_changes=true 强制删除，"
                        "或用 keep_worktree 保留以供审查。")
        ok1, _ = self.run_git(["worktree", "remove", str(path), "--force"])
        if not ok1:
            return f"删除 worktree '{name}' 目录失败"
        self.run_git(["branch", "-D", f"wt/{name}"])
        self.log_event("remove", name)
        print(f"  \033[33m[worktree] removed: {name}\033[0m")
        return f"Worktree '{name}' removed"

    def keep_worktree(self, name: str) -> str:
        """保留 worktree 供人工审查，分支不删。"""
        err = self.validate_worktree_name(name)
        if err:
            return err
        self.log_event("keep", name)
        print(f"  \033[36m[worktree] kept: {name}\033[0m")
        return f"Worktree '{name}' kept for review (branch: wt/{name})"

    def worktree_cwd(self, task_id: str) -> Path | None:
        """返回任务绑定的 worktree 目录，供队友工具作为 cwd 使用。"""
        try:
            task = self.tasks.load_task(task_id)
        except FileNotFoundError:
            return None
        if task.worktree:
            return self.worktrees_dir / task.worktree
        return None

    # ── OpenAI 工具处理器（参数为 arguments dict）──

    def run_create_worktree(self, arguments) -> str:
        error = require_param(arguments, "name")
        if error:
            return error
        return self.create_worktree(arguments["name"], arguments.get("task_id", ""))

    def run_remove_worktree(self, arguments) -> str:
        error = require_param(arguments, "name")
        if error:
            return error
        return self.remove_worktree(
            arguments["name"], bool(arguments.get("discard_changes", False)))

    def run_keep_worktree(self, arguments) -> str:
        error = require_param(arguments, "name")
        if error:
            return error
        return self.keep_worktree(arguments["name"])

    def register_tools(self, agent_tool):
        agent_tool.register_tool(
            name="create_worktree",
            description="创建一个隔离的 git worktree（独立分支），可选绑定到任务。",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "worktree 名称"},
                    "task_id": {"type": "string", "description": "可选：要绑定的任务 ID"},
                },
                "required": ["name"],
            },
            func=self.run_create_worktree,
        )
        agent_tool.register_tool(
            name="remove_worktree",
            description="删除 worktree。若有未提交改动则拒绝，除非 discard_changes=true。",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "worktree 名称"},
                    "discard_changes": {"type": "boolean",
                                        "description": "true=强制删除并丢弃改动"},
                },
                "required": ["name"],
            },
            func=self.run_remove_worktree,
        )
        agent_tool.register_tool(
            name="keep_worktree",
            description="保留 worktree 供人工审查，分支不删除。",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "worktree 名称"},
                },
                "required": ["name"],
            },
            func=self.run_keep_worktree,
        )