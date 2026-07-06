"""Agent 任务管理：文件持久化的任务图 + blockedBy 依赖 + OpenAI 工具注册。

与原版差异：``tasks_dir`` 必须显式传入（不再从全局 config 默认）。
"""

import json
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from xiaozhi.tool_utils import require_param


@dataclass
class Task:
    id: str
    subject: str
    description: str
    status: str          # pending | in_progress | completed
    owner: str | None
    blockedBy: list[str]
    worktree: str | None = None   # 绑定的 worktree 名称


class AgentTask:
    """Agent 的任务管理类：文件持久化的任务图 + blockedBy 依赖 + OpenAI 工具注册。"""

    def __init__(self, tasks_dir: Path):
        self.tasks_dir = Path(tasks_dir)
        self.tasks_dir.mkdir(parents=True, exist_ok=True)

    def _task_path(self, task_id: str) -> Path:
        return self.tasks_dir / f"{task_id}.json"

    def save_task(self, task: Task):
        self._task_path(task.id).write_text(
            json.dumps(asdict(task), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def load_task(self, task_id: str) -> Task:
        raw = self._task_path(task_id).read_text(encoding="utf-8")
        return Task(**json.loads(raw))

    def list_tasks(self) -> list[Task]:
        return [Task(**json.loads(p.read_text(encoding="utf-8")))
                for p in sorted(self.tasks_dir.glob("task_*.json"))]

    def create_task(self, subject: str, description: str = "",
                    blockedBy: list[str] | None = None) -> Task:
        task = Task(
            id=f"task_{int(time.time())}_{random.randint(0, 9999):04d}",
            subject=subject,
            description=description,
            status="pending",
            owner=None,
            blockedBy=blockedBy or [],
        )
        self.save_task(task)
        return task

    def get_task(self, task_id: str) -> str:
        task = self.load_task(task_id)
        return json.dumps(asdict(task), indent=2, ensure_ascii=False)

    def can_start(self, task_id: str) -> bool:
        task = self.load_task(task_id)
        for dep_id in task.blockedBy:
            if not self._task_path(dep_id).exists():
                return False
            if self.load_task(dep_id).status != "completed":
                return False
        return True

    def _pending_deps(self, task: Task) -> list[str]:
        return [d for d in task.blockedBy
                if not self._task_path(d).exists()
                or self.load_task(d).status != "completed"]

    def claim_task(self, task_id: str, owner: str = "agent") -> str:
        task = self.load_task(task_id)
        if task.status != "pending":
            return f"任务 {task_id} 处于 {task.status} 状态，无法认领"
        if not self.can_start(task_id):
            return f"被以下前置任务阻塞：{self._pending_deps(task)}"
        task.owner = owner
        task.status = "in_progress"
        self.save_task(task)
        print(f"  \033[36m[claim] {task.subject} → in_progress (owner: {owner})\033[0m")
        return f"已认领 {task.id} ({task.subject})"

    def complete_task(self, task_id: str) -> str:
        task = self.load_task(task_id)
        if task.status != "in_progress":
            return f"任务 {task_id} 处于 {task.status} 状态，无法完成"
        task.status = "completed"
        self.save_task(task)
        unblocked = [t.subject for t in self.list_tasks()
                     if t.status == "pending" and t.blockedBy and self.can_start(t.id)]
        print(f"  \033[32m[complete] {task.subject} ✓\033[0m")
        msg = f"已完成 {task.id} ({task.subject})"
        if unblocked:
            msg += f"\n已解锁：{', '.join(unblocked)}"
            print(f"  \033[33m[unblocked] {', '.join(unblocked)}\033[0m")
        return msg

    def _require_task_id(self, arguments) -> tuple[str | None, str | None]:
        """Return (task_id, error_message)."""
        task_id = arguments.get("task_id")
        if not task_id:
            return None, "Error: 缺少参数 task_id"
        return task_id, None

    def _run_with_task_id(self, arguments, handler) -> str:
        task_id, error = self._require_task_id(arguments)
        if error:
            return error
        try:
            return handler(task_id)
        except FileNotFoundError:
            return f"Error: 任务 {task_id} 不存在"

    def run_create_task(self, arguments) -> str:
        error = require_param(arguments, "subject")
        if error:
            return error
        blockedBy = arguments.get("blockedBy") or []
        task = self.create_task(arguments["subject"], arguments.get("description", ""), blockedBy)
        deps = f"（blockedBy: {', '.join(blockedBy)}）" if blockedBy else ""
        print(f"  \033[34m[create] {task.subject}{deps}\033[0m")
        return f"已创建 {task.id}: {task.subject}{deps}"

    def run_list_tasks(self, arguments=None) -> str:
        tasks = self.list_tasks()
        if not tasks:
            return "暂无任务。可用 create_task 创建。"
        lines = []
        for t in tasks:
            icon = {"pending": "○", "in_progress": "●",
                    "completed": "✓"}.get(t.status, "?")
            deps = f"（blockedBy: {', '.join(t.blockedBy)}）" if t.blockedBy else ""
            owner = f" [{t.owner}]" if t.owner else ""
            lines.append(f"  {icon} {t.id}: {t.subject} [{t.status}]{owner}{deps}")
        return "\n".join(lines)

    def run_get_task(self, arguments) -> str:
        return self._run_with_task_id(arguments, self.get_task)

    def run_claim_task(self, arguments) -> str:
        return self._run_with_task_id(
            arguments,
            lambda tid: self.claim_task(tid, owner=arguments.get("owner", "agent")),
        )

    def run_complete_task(self, arguments) -> str:
        return self._run_with_task_id(arguments, self.complete_task)

    def register_tools(self, agent_tool):
        agent_tool.register_tool(
            name="create_task",
            description="创建一个新任务，可选 blockedBy 依赖。",
            parameters={
                "type": "object",
                "properties": {
                    "subject": {"type": "string", "description": "任务标题"},
                    "description": {"type": "string", "description": "任务详细描述"},
                    "blockedBy": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "依赖的前置任务 ID 列表",
                    },
                },
                "required": ["subject"],
            },
            func=self.run_create_task,
        )
        agent_tool.register_tool(
            name="list_tasks",
            description="列出所有任务及其状态、负责人和依赖。",
            parameters={"type": "object", "properties": {}, "required": []},
            func=self.run_list_tasks,
        )
        agent_tool.register_tool(
            name="get_task",
            description="按 ID 获取某个任务的完整详情。",
            parameters={
                "type": "object",
                "properties": {"task_id": {"type": "string", "description": "任务 ID"}},
                "required": ["task_id"],
            },
            func=self.run_get_task,
        )
        agent_tool.register_tool(
            name="claim_task",
            description="认领一个 pending 任务，设置 owner 并置为 in_progress。",
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "任务 ID"},
                    "owner": {"type": "string", "description": "负责人名称，默认 agent"},
                },
                "required": ["task_id"],
            },
            func=self.run_claim_task,
        )
        agent_tool.register_tool(
            name="complete_task",
            description="完成一个 in_progress 任务，并报告被解锁的下游任务。",
            parameters={
                "type": "object",
                "properties": {"task_id": {"type": "string", "description": "任务 ID"}},
                "required": ["task_id"],
            },
            func=self.run_complete_task,
        )