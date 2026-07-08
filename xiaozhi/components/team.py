"""多 Agent 协作：teammate + 基于 jsonl 收件箱的消息总线 + 团队协议。

与 ``orchestration.py`` 的 ``AgentTeam``（Lead 一次性分解 → Worker 并行 → 聚合）不同，
本模块提供**长期存活的 teammate**：Lead 派生后台 teammate，双方通过文件收件箱异步通信，
teammate 空闲时自动从任务板认领任务，支持 shutdown / plan_approval 请求-响应协议。

范式对比：
- ``AgentTeam``  : fan-out / fan-in，一次性、同步聚合，适合可并行分解的单个问题。
- ``Team``（本模块）: 长期 teammate + 消息总线 + 自动认领，适合持续协作的多角色团队。

依赖注入：``client`` / ``model`` / ``workdir`` / ``tasks`` / ``worktree`` 通过构造函数传入，
不引用全局。收件箱落 ``workdir/.mailboxes/<agent>.jsonl``。

用法::

    from xiaozhi.components.team import Team
    team = Team(client, model, workdir, tasks, worktree)
    team.register_tools(agent_tool)         # 给 Lead 注册协作工具
    team.spawn_teammate("frontend", "前端工程师", "实现登录页")
    ...
    for msg in team.consume_lead_inbox():   # 主循环消费 Lead 收件箱
        ...
"""

import json
import subprocess
import threading
import time
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Optional

from xiaozhi.llm import chat_completion_stream
from xiaozhi.context_manager import AgentContextManager
from xiaozhi.tool_runner import run_tool_calls

# 空闲循环参数
IDLE_POLL_INTERVAL = 5    # 秒
IDLE_TIMEOUT = 60         # 秒


class MessageBus:
    """基于文件的消息总线。每个 Agent 一个 jsonl 收件箱。

    读取是消费式的（read + unlink，读完即删）。教学版本不做文件锁；
    生产环境应使用 proper-lockfile 等保证并发安全。"""

    def __init__(self, mailbox_dir: Path):
        self.mailbox_dir = Path(mailbox_dir)
        self.mailbox_dir.mkdir(parents=True, exist_ok=True)

    def send(self, from_agent: str, to_agent: str, content: str,
             msg_type: str = "message", metadata: Optional[dict] = None):
        msg = {"from": from_agent, "to": to_agent, "content": content,
               "type": msg_type, "ts": time.time(), "metadata": metadata or {}}
        inbox = self.mailbox_dir / f"{to_agent}.jsonl"
        with open(inbox, "a", encoding="utf-8") as f:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")
        print(f"  \033[33m[bus] {from_agent} → {to_agent}: "
              f"({msg_type}) {content[:50]}\033[0m")

    def read_inbox(self, agent: str) -> list[dict]:
        inbox = self.mailbox_dir / f"{agent}.jsonl"
        if not inbox.exists():
            return []
        msgs = [json.loads(line)
                for line in inbox.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        inbox.unlink()  # 消费式：读完删除
        return msgs

    def peek(self, agent: str) -> bool:
        """非消费式：判断该 Agent 是否有未读消息。"""
        inbox = self.mailbox_dir / f"{agent}.jsonl"
        return inbox.exists() and inbox.stat().st_size > 0


@dataclass
class ProtocolState:
    """团队协议的请求状态（shutdown / plan_approval）。"""
    request_id: str
    type: str       # "shutdown" | "plan_approval"
    sender: str
    target: str
    status: str     # pending | approved | rejected
    payload: str    # plan 文本或 shutdown 原因
    created_at: float = field(default_factory=time.time)


# teammate 可用的工具（OpenAI 兼容格式）
SUB_TOOLS = [
    {"type": "function", "function": {
        "name": "bash", "description": "执行一条 shell 命令。",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "读取文件内容。",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "写入内容到文件。",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "send_message", "description": "给另一个 Agent 发送消息。",
        "parameters": {"type": "object",
                       "properties": {"to": {"type": "string"},
                                      "content": {"type": "string"}},
                       "required": ["to", "content"]}}},
    {"type": "function", "function": {
        "name": "submit_plan", "description": "提交计划给 Lead 审批。",
        "parameters": {"type": "object",
                       "properties": {"plan": {"type": "string"}},
                       "required": ["plan"]}}},
    {"type": "function", "function": {
        "name": "list_tasks", "description": "列出任务板上所有任务。",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "claim_task", "description": "认领一个 pending 任务。",
        "parameters": {"type": "object",
                       "properties": {"task_id": {"type": "string"}},
                       "required": ["task_id"]}}},
    {"type": "function", "function": {
        "name": "complete_task", "description": "完成一个 in_progress 任务。",
        "parameters": {"type": "object",
                       "properties": {"task_id": {"type": "string"}},
                       "required": ["task_id"]}}},
]


class Team:
    """长期存活 teammate 的协作团队：消息总线 + 自动认领 + 团队协议。

    零全局状态：所有共享状态（收件箱、活跃 teammate、协议请求）都收在实例里。
    """

    def __init__(self, client, model: str, workdir, tasks, worktree=None):
        self.client = client
        self.model = model
        self.workdir = Path(workdir)
        self.tasks = tasks          # AgentTask 实例（文件持久化任务图）
        self.worktree = worktree    # AgentWorktree 实例（可选，任务隔离目录）
        self.bus = MessageBus(self.workdir / ".mailboxes")
        self.active_teammates: dict[str, bool] = {}
        self.pending_requests: dict[str, ProtocolState] = {}

    # ── 协议 ──────────────────────────────────────────────

    @staticmethod
    def _new_request_id() -> str:
        return f"req_{random.randint(0, 999999):06d}"

    def _match_response(self, response_type: str, request_id: str, approve: bool):
        """通过 request_id 把响应关联回原始请求，并校验类型匹配。"""
        state = self.pending_requests.get(request_id)
        if not state:
            print(f"  \033[31m[protocol] unknown request_id: {request_id}\033[0m")
            return
        expected = {"shutdown": "shutdown_response",
                    "plan_approval": "plan_approval_response"}.get(state.type)
        if expected and response_type != expected:
            print(f"  \033[31m[protocol] type mismatch: expected {expected}, "
                  f"got {response_type}\033[0m")
            return
        if state.status != "pending":
            print(f"  \033[33m[protocol] {request_id} already {state.status}, "
                  f"ignoring duplicate\033[0m")
            return
        state.status = "approved" if approve else "rejected"
        icon = "✓" if approve else "✗"
        color = "32" if approve else "31"
        print(f"  \033[{color}m[protocol] {state.type} {icon} "
              f"({request_id}: {state.status})\033[0m")

    def consume_lead_inbox(self, route_protocol: bool = True) -> list[dict]:
        """读取 Lead 收件箱；路由协议响应后返回全部消息。

        主循环每轮调用它，把 teammate 发回的 result / 协议响应注入对话。"""
        msgs = self.bus.read_inbox("lead")
        if not msgs:
            return []
        if route_protocol:
            for msg in msgs:
                meta = msg.get("metadata", {})
                req_id = meta.get("request_id", "")
                msg_type = msg.get("type", "")
                if req_id and msg_type.endswith("_response"):
                    self._match_response(msg_type, req_id, meta.get("approve", False))
        return msgs

    # ── 任务板自动认领 ────────────────────────────────────

    def _scan_unclaimed_tasks(self) -> list[dict]:
        """扫描 pending、无 owner、且依赖已全部完成的任务。"""
        unclaimed = []
        for f in sorted(self.tasks.tasks_dir.glob("task_*.json")):
            task = json.loads(f.read_text(encoding="utf-8"))
            if (task.get("status") == "pending"
                    and not task.get("owner")
                    and self.tasks.can_start(task["id"])):
                unclaimed.append(task)
        return unclaimed

    def _idle_poll(self, name: str, messages: list,
                   wt_ctx: Optional[dict] = None) -> str:
        """空闲轮询最多 IDLE_TIMEOUT 秒。

        返回 'work'（有新活，注入 messages 后回 LLM 轮）、
        'shutdown'（收到 shutdown_request）、'timeout'（无事可做，应退出）。"""
        for _ in range(IDLE_TIMEOUT // IDLE_POLL_INTERVAL):
            time.sleep(IDLE_POLL_INTERVAL)

            # 1) 先看收件箱 —— 协议消息优先
            inbox = self.bus.read_inbox(name)
            if inbox:
                for msg in inbox:
                    if msg.get("type") == "shutdown_request":
                        req_id = msg.get("metadata", {}).get("request_id", "")
                        self.bus.send(name, "lead", "Shutting down gracefully.",
                                      "shutdown_response",
                                      {"request_id": req_id, "approve": True})
                        print(f"  \033[35m[protocol] {name} approved shutdown "
                              f"in idle ({req_id})\033[0m")
                        return "shutdown"
                messages.append({"role": "user",
                                 "content": "<inbox>" + json.dumps(inbox, ensure_ascii=False) + "</inbox>"})
                print(f"  \033[36m[idle] {name} found inbox messages\033[0m")
                return "work"

            # 2) 扫描任务板 —— 自动认领可执行的任务
            unclaimed = self._scan_unclaimed_tasks()
            if unclaimed:
                task = unclaimed[0]
                result = self.tasks.claim_task(task["id"], owner=name)
                if "已认领" in result or "Claimed" in result:
                    wt_info = ""
                    if wt_ctx is not None and self.worktree is not None:
                        wt_ctx["path"] = self.worktree.worktree_cwd(task["id"])
                        if wt_ctx["path"]:
                            wt_info = f"\n工作目录: {wt_ctx['path']}"
                            print(f"  \033[33m[worktree] {name} → {wt_ctx['path']}\033[0m")
                    messages.append({"role": "user",
                                     "content": f"<auto-claimed>Task {task['id']}: "
                                                f"{task['subject']}{wt_info}</auto-claimed>"})
                    print(f"  \033[32m[idle] {name} auto-claimed: {task['subject']}\033[0m")
                    return "work"
                print(f"  \033[33m[idle] {name} claim failed: {result}\033[0m")

        print(f"  \033[31m[idle] {name} timeout ({IDLE_TIMEOUT}s)\033[0m")
        return "timeout"

    # ── teammate 工具执行 ─────────────────────────────────

    def _run_bash(self, command: str, cwd: Optional[Path] = None) -> str:
        try:
            r = subprocess.run(command, shell=True, cwd=cwd or self.workdir,
                               capture_output=True, text=True, timeout=120)
            out = (r.stdout + r.stderr).strip()
            return out[:50000] if out else "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: Timeout (120s)"

    def _run_read(self, path: str, cwd: Optional[Path] = None) -> str:
        try:
            return ((cwd or self.workdir) / path).read_text(encoding="utf-8")
        except Exception as e:
            return f"Error: {e}"

    def _run_write(self, path: str, content: str, cwd: Optional[Path] = None) -> str:
        try:
            fp = (cwd or self.workdir) / path
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(content, encoding="utf-8")
            return f"Wrote {len(content)} bytes to {path}"
        except Exception as e:
            return f"Error: {e}"

    def _teammate_submit_plan(self, from_name: str, plan: str) -> str:
        """teammate 向 Lead 提交计划请求审批（协议层请求，非代码层门禁）。"""
        req_id = self._new_request_id()
        self.pending_requests[req_id] = ProtocolState(
            request_id=req_id, type="plan_approval",
            sender=from_name, target="lead", status="pending", payload=plan)
        self.bus.send(from_name, "lead", plan, "plan_approval_request",
                      {"request_id": req_id})
        return f"Plan submitted ({req_id}). Waiting for approval..."

    # ── teammate 后台线程 ─────────────────────────────────

    def spawn_teammate(self, name: str, role: str, prompt: str) -> str:
        """在后台线程派生一个 teammate agent（WORK → IDLE → SHUTDOWN 生命周期）。"""
        if name in self.active_teammates:
            return f"Teammate '{name}' 已存在"

        system = (f"你是 '{name}'，一位 {role}。使用工具完成任务，"
                  f"通过 send_message 把结果发给 'lead'。"
                  f"你可以用 list_tasks/claim_task/complete_task 从任务板认领任务。"
                  f"若任务绑定了 worktree，你的 bash/读写都会自动在该隔离目录中执行。"
                  f"注意收件箱里的协议消息（shutdown_request 等）。")

        wt_ctx: dict[str, Optional[Path]] = {"path": None}

        def sub_executor(tool_name: str, arguments: dict) -> str:
            cwd = wt_ctx["path"]
            if tool_name == "bash":
                return self._run_bash(arguments.get("command", ""), cwd=cwd)
            if tool_name == "read_file":
                return self._run_read(arguments.get("path", ""), cwd=cwd)
            if tool_name == "write_file":
                return self._run_write(arguments.get("path", ""),
                                       arguments.get("content", ""), cwd=cwd)
            if tool_name == "send_message":
                self.bus.send(name, arguments.get("to", ""), arguments.get("content", ""))
                return "Sent"
            if tool_name == "submit_plan":
                return self._teammate_submit_plan(name, arguments.get("plan", ""))
            if tool_name == "list_tasks":
                return self.tasks.run_list_tasks({})
            if tool_name == "claim_task":
                task_id = arguments.get("task_id")
                result = self.tasks.run_claim_task({"task_id": task_id, "owner": name})
                if ("已认领" in result or "Claimed" in result) and self.worktree is not None:
                    wt_ctx["path"] = self.worktree.worktree_cwd(task_id)
                    if wt_ctx["path"]:
                        print(f"  \033[33m[worktree] {name} → {wt_ctx['path']}\033[0m")
                return result
            if tool_name == "complete_task":
                result = self.tasks.run_complete_task({"task_id": arguments.get("task_id")})
                wt_ctx["path"] = None
                return result
            return f"Unknown tool: {tool_name}"

        def handle_inbox_message(msg: dict, messages: list) -> bool:
            """按 type 分派协议消息。返回 True 表示 teammate 应停止。"""
            msg_type = msg.get("type", "message")
            meta = msg.get("metadata", {})
            req_id = meta.get("request_id", "")
            if msg_type == "shutdown_request":
                self.bus.send(name, "lead", "Shutting down gracefully.",
                              "shutdown_response", {"request_id": req_id, "approve": True})
                print(f"  \033[35m[protocol] {name} approved shutdown ({req_id})\033[0m")
                return True
            if msg_type == "plan_approval_response":
                if meta.get("approve", False):
                    messages.append({"role": "user",
                                     "content": "[Plan approved] Proceed with the task."})
                else:
                    messages.append({"role": "user",
                                     "content": f"[Plan rejected] Feedback: {msg['content']}"})
            return False

        def drain_inbox(messages: list) -> bool:
            """读收件箱：协议消息走 dispatch，其余注入 <inbox>。返回 True 表示应停止。"""
            inbox = self.bus.read_inbox(name)
            non_protocol = []
            for msg in inbox:
                if msg.get("type") in ("shutdown_request", "plan_approval_response"):
                    if handle_inbox_message(msg, messages):
                        return True
                else:
                    non_protocol.append(msg)
            if non_protocol:
                messages.append({"role": "user",
                                 "content": "<inbox>" + json.dumps(non_protocol, ensure_ascii=False) + "</inbox>"})
            return False

        def run():
            ctx = AgentContextManager(self.client, self.model, self.workdir)
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": prompt}]
            summary = "Done."
            shutdown = False

            while not shutdown:
                # 身份重注入：上下文压缩到很短时，重新提醒身份和角色
                if len(messages) <= 3:
                    messages.insert(1, {"role": "user",
                                        "content": f"<identity>你是 '{name}'，角色：{role}。"
                                                   f"继续你的工作。</identity>"})

                # WORK 阶段：最多 10 轮
                for _ in range(10):
                    if drain_inbox(messages):
                        shutdown = True
                        break
                    try:
                        llm_response, tool_calls, _ = chat_completion_stream(
                            client=self.client,
                            messages=ctx.snip_compact(messages, max_messages=20),
                            tools=SUB_TOOLS, model=self.model, print_output=False)
                    except Exception:
                        shutdown = True
                        break
                    if llm_response:
                        summary = llm_response
                    if not tool_calls:
                        break  # 无工具调用 → 进入 IDLE
                    messages.append({"role": "assistant",
                                     "content": llm_response or None,
                                     "tool_calls": tool_calls})
                    messages = run_tool_calls(tool_calls, sub_executor, messages)

                if shutdown:
                    break

                # IDLE 阶段：轮询收件箱 + 自动认领任务
                idle_result = self._idle_poll(name, messages, wt_ctx)
                if idle_result in ("shutdown", "timeout"):
                    break
                # "work" → 回到 WORK 阶段

            self.bus.send(name, "lead", summary, "result")
            self.active_teammates.pop(name, None)
            print(f"  \033[32m[teammate] {name} finished\033[0m")

        self.active_teammates[name] = True
        threading.Thread(target=run, daemon=True).start()
        print(f"  \033[36m[teammate] {name} spawned as {role}\033[0m")
        return f"Teammate '{name}' spawned as {role}"

    # ── Lead 侧协议/通信工具 ──────────────────────────────

    def request_shutdown(self, teammate: str) -> str:
        req_id = self._new_request_id()
        self.pending_requests[req_id] = ProtocolState(
            request_id=req_id, type="shutdown",
            sender="lead", target=teammate, status="pending", payload="")
        self.bus.send("lead", teammate, "Please shut down gracefully.",
                      "shutdown_request", {"request_id": req_id})
        print(f"  \033[35m[protocol] shutdown_request → {teammate} ({req_id})\033[0m")
        return f"Shutdown request sent to {teammate} (req: {req_id})"

    def request_plan(self, teammate: str, task: str) -> str:
        """Lead 要求 teammate 为某任务提交计划。"""
        self.bus.send("lead", teammate, f"Please submit a plan for: {task}", "message")
        return f"Asked {teammate} to submit a plan"

    def review_plan(self, request_id: str, approve: bool, feedback: str = "") -> str:
        state = self.pending_requests.get(request_id)
        if not state:
            return f"Request {request_id} not found"
        if state.status != "pending":
            return f"Request {request_id} already {state.status}"
        state.status = "approved" if approve else "rejected"
        self.bus.send("lead", state.sender,
                      feedback or ("Approved" if approve else "Rejected"),
                      "plan_approval_response",
                      {"request_id": request_id, "approve": approve})
        icon = "✓" if approve else "✗"
        print(f"  \033[32m[protocol] plan {icon} ({request_id})\033[0m")
        return f"Plan {'approved' if approve else 'rejected'} ({request_id})"

    def send_message(self, to: str, content: str) -> str:
        self.bus.send("lead", to, content)
        return f"Sent to {to}"

    def check_inbox(self) -> str:
        """检查 Lead 收件箱，自动路由协议响应。"""
        msgs = self.consume_lead_inbox(route_protocol=True)
        if not msgs:
            return "(inbox empty)"
        lines = []
        for m in msgs:
            meta = m.get("metadata", {})
            req_id = meta.get("request_id", "")
            tag = f" [{m['type']} req:{req_id}]" if req_id else f" [{m['type']}]"
            lines.append(f"  [{m['from']}]{tag} {m['content'][:200]}")
        return "\n".join(lines)

    # ── 给 Lead 的 AgentTool 注册协作工具 ─────────────────

    def register_tools(self, agent_tool):
        """把 spawn_teammate / send_message / check_inbox / 协议工具注册给 Lead。"""
        agent_tool.register_tool(
            name="spawn_teammate",
            description="派生一个后台 teammate agent（如 frontend/backend 工程师）去执行任务。",
            parameters={"type": "object", "properties": {
                "name": {"type": "string", "description": "teammate 名称，如 frontend"},
                "role": {"type": "string", "description": "角色，如 前端工程师"},
                "prompt": {"type": "string", "description": "交给该 teammate 的初始任务描述"},
            }, "required": ["name", "role", "prompt"]},
            func=lambda a: self.spawn_teammate(a.get("name", ""), a.get("role", ""),
                                               a.get("prompt", "")))
        agent_tool.register_tool(
            name="send_message",
            description="给某个 teammate 发送一条消息（进入其收件箱）。",
            parameters={"type": "object", "properties": {
                "to": {"type": "string", "description": "接收方 teammate 名称"},
                "content": {"type": "string", "description": "消息内容"},
            }, "required": ["to", "content"]},
            func=lambda a: self.send_message(a.get("to", ""), a.get("content", "")))
        agent_tool.register_tool(
            name="check_inbox",
            description="主动查看 Lead 收件箱（会自动路由协议响应）。",
            parameters={"type": "object", "properties": {}, "required": []},
            func=lambda a: self.check_inbox())
        agent_tool.register_tool(
            name="request_shutdown",
            description="请求某个 teammate 优雅关闭。",
            parameters={"type": "object", "properties": {
                "teammate": {"type": "string", "description": "teammate 名称"},
            }, "required": ["teammate"]},
            func=lambda a: self.request_shutdown(a.get("teammate", "")))
        agent_tool.register_tool(
            name="request_plan",
            description="要求某个 teammate 先为某任务提交计划再执行。",
            parameters={"type": "object", "properties": {
                "teammate": {"type": "string", "description": "teammate 名称"},
                "task": {"type": "string", "description": "任务描述"},
            }, "required": ["teammate", "task"]},
            func=lambda a: self.request_plan(a.get("teammate", ""), a.get("task", "")))
        agent_tool.register_tool(
            name="review_plan",
            description="审批某个 teammate 提交的计划（通过 request_id）。",
            parameters={"type": "object", "properties": {
                "request_id": {"type": "string", "description": "计划请求 ID"},
                "approve": {"type": "boolean", "description": "是否通过"},
                "feedback": {"type": "string", "description": "审批意见（可选）"},
            }, "required": ["request_id", "approve"]},
            func=lambda a: self.review_plan(a.get("request_id", ""),
                                            a.get("approve", False),
                                            a.get("feedback", "")))
