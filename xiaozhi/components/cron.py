"""Cron 调度器（OpenAI 兼容格式）。

与原版差异：``durable_path`` 必须显式传入，不再从全局 WORKDIR 派生。

四层结构：
  1. 调度线程：cron_scheduler_loop 每秒轮询，命中的 job 入队 cron_queue
  2. 队列：cron_queue 解耦调度线程与 agent loop
  3. 队列处理线程：queue_processor_loop 在 agent 空闲时投递排队的 job
  4. 消费：agent loop 消费 cron_queue，把 prompt 注入 messages
"""

import json
import random
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from xiaozhi.tool_utils import require_param


@dataclass
class CronJob:
    id: str
    cron: str        # 五段式 cron 表达式，如 "0 9 * * *"
    prompt: str      # 触发时注入给 Agent 的消息
    recurring: bool  # True=周期性，False=一次性
    durable: bool    # True=写磁盘，跨会话保留


def _cron_field_matches(field: str, value: int) -> bool:
    """匹配单个 cron 字段与具体值。支持 * / */n / a,b,c / a-b / 数字。"""
    if field == "*":
        return True
    if field.startswith("*/"):
        step = int(field[2:])
        return step > 0 and value % step == 0
    if "," in field:
        return any(_cron_field_matches(f.strip(), value)
                   for f in field.split(","))
    if "-" in field:
        lo, hi = field.split("-", 1)
        return int(lo) <= value <= int(hi)
    return value == int(field)


def _validate_cron_field(field: str, lo: int, hi: int) -> str | None:
    """校验单个 cron 字段是否落在 [lo, hi] 内。返回错误信息或 None。"""
    if field == "*":
        return None
    if field.startswith("*/"):
        step_str = field[2:]
        if not step_str.isdigit():
            return f"步长非法：{field}"
        if int(step_str) <= 0:
            return f"步长必须 > 0：{field}"
        return None
    if "," in field:
        for part in field.split(","):
            err = _validate_cron_field(part.strip(), lo, hi)
            if err:
                return err
        return None
    if "-" in field:
        parts = field.split("-", 1)
        if not parts[0].isdigit() or not parts[1].isdigit():
            return f"区间非法：{field}"
        a, b = int(parts[0]), int(parts[1])
        if a < lo or a > hi or b < lo or b > hi:
            return f"区间 {field} 超出范围 [{lo}-{hi}]"
        if a > b:
            return f"区间起点大于终点：{field}"
        return None
    if not field.isdigit():
        return f"字段非法：{field}"
    val = int(field)
    if val < lo or val > hi:
        return f"值 {val} 超出范围 [{lo}-{hi}]"
    return None


def validate_cron(cron_expr: str) -> str | None:
    """校验五段式 cron 表达式。返回错误信息或 None（表示合法）。"""
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return f"期望 5 个字段，实际 {len(fields)} 个"
    bounds = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 6)]
    names = ["minute", "hour", "day-of-month", "month", "day-of-week"]
    for field, (lo, hi), name in zip(fields, bounds, names):
        err = _validate_cron_field(field, lo, hi)
        if err:
            return f"{name}: {err}"
    return None


def cron_matches(cron_expr: str, dt: datetime) -> bool:
    """判断五段式 cron 是否匹配给定时间。
    标准语义：分钟、小时、月必须匹配；日和星期任一匹配（两者都被限定时取 OR）。"""
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return False
    minute, hour, day, month, week = fields
    week_val = (dt.weekday() + 1) % 7  # Python 周一=0 → cron 周日=0

    m = _cron_field_matches(minute, dt.minute)
    h = _cron_field_matches(hour, dt.hour)
    _day = _cron_field_matches(day, dt.day)
    _month = _cron_field_matches(month, dt.month)
    _week = _cron_field_matches(week, week_val)

    if not (m and h and _month):
        return False
    day_unconstrained = day == "*"
    week_unconstrained = week == "*"
    if day_unconstrained and week_unconstrained:
        return True
    if day_unconstrained:
        return _week
    if week_unconstrained:
        return _day
    return _day or _week


class AgentCron:
    """Agent 的 cron 调度器：五段式表达式匹配 + 文件持久化 + 独立调度线程 + OpenAI 工具注册。"""

    def __init__(self, durable_path: Path):
        self.durable_path = Path(durable_path)
        self.scheduled_jobs: dict[str, CronJob] = {}
        self.cron_queue: list[CronJob] = []
        self.cron_lock = threading.Lock()
        self._last_fired: dict[str, str] = {}  # job_id → "YYYY-MM-DD HH:MM"
        self._scheduler_started = False
        self.load_durable_jobs()

    # ── 持久化 ──

    def save_durable_jobs(self):
        """把 durable job 持久化到 .scheduled_tasks.json。"""
        durable = [asdict(j) for j in self.scheduled_jobs.values() if j.durable]
        self.durable_path.write_text(
            json.dumps(durable, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def load_durable_jobs(self):
        """启动时从磁盘加载 durable job。"""
        if not self.durable_path.exists():
            return
        try:
            jobs = json.loads(self.durable_path.read_text(encoding="utf-8"))
        except Exception:
            return
        loaded = 0
        for j in jobs:
            try:
                job = CronJob(**j)
            except TypeError:
                continue
            err = validate_cron(job.cron)
            if err:
                print(f"  \033[31m[cron] 跳过非法 job {job.id}: {err}\033[0m")
                continue
            self.scheduled_jobs[job.id] = job
            loaded += 1
        if loaded:
            print(f"  \033[35m[cron] 已加载 {loaded} 个 durable job\033[0m")

    # ── 注册 / 取消 ──

    def schedule_job(self, cron: str, prompt: str, recurring: bool = True,
                     durable: bool = True) -> CronJob | str:
        """校验并注册一个 cron job。返回 CronJob 或错误字符串。"""
        err = validate_cron(cron)
        if err:
            return err
        job = CronJob(
            id=f"cron_{random.randint(0, 999999):06d}",
            cron=cron, prompt=prompt,
            recurring=recurring, durable=durable,
        )
        with self.cron_lock:
            self.scheduled_jobs[job.id] = job
        if durable:
            self.save_durable_jobs()
        print(f"  \033[35m[cron register] {job.id} '{cron}' → {prompt[:40]}\033[0m")
        return job

    def cancel_job(self, job_id: str) -> str:
        """取消一个 cron job。"""
        with self.cron_lock:
            job = self.scheduled_jobs.pop(job_id, None)
        if not job:
            return f"未找到 job {job_id}"
        if job.durable:
            self.save_durable_jobs()
        print(f"  \033[31m[cron cancel] {job_id}\033[0m")
        return f"已取消 {job_id}"

    # ── 队列 ──

    def consume_cron_queue(self) -> list[CronJob]:
        """消费已触发的 job（由 agent loop 调用）。"""
        with self.cron_lock:
            fired = list(self.cron_queue)
            self.cron_queue.clear()
        return fired

    def has_cron_queue(self) -> bool:
        """是否有已触发、等待投递的 job。"""
        with self.cron_lock:
            return bool(self.cron_queue)

    # ── 调度线程 ──

    def cron_scheduler_loop(self):
        """独立守护线程：每秒轮询，命中的 job 入队。"""
        while True:
            time.sleep(1)
            now = datetime.now()
            # 带日期的分钟标记，防止跨天时日级任务被漏发
            minute_marker = now.strftime("%Y-%m-%d %H:%M")
            with self.cron_lock:
                for job in list(self.scheduled_jobs.values()):
                    try:
                        if cron_matches(job.cron, now):
                            if self._last_fired.get(job.id) != minute_marker:
                                self.cron_queue.append(job)
                                self._last_fired[job.id] = minute_marker
                                print(f"  \033[35m[cron fire] {job.id} → "
                                      f"{job.prompt[:40]}\033[0m")
                            if not job.recurring:
                                self.scheduled_jobs.pop(job.id, None)
                                if job.durable:
                                    self.save_durable_jobs()
                    except Exception as e:
                        print(f"  \033[31m[cron error] {job.id}: {e}\033[0m")

    def start_scheduler(self):
        """启动调度守护线程（幂等）。"""
        if self._scheduler_started:
            return
        self._scheduler_started = True
        threading.Thread(target=self.cron_scheduler_loop, daemon=True).start()
        print("  \033[35m[cron] scheduler thread started\033[0m")

    # ── 工具处理函数（func(arguments) 签名）──

    def run_schedule_cron(self, arguments) -> str:
        err = require_param(arguments, "cron") or require_param(arguments, "prompt")
        if err:
            return err
        result = self.schedule_job(
            arguments["cron"],
            arguments["prompt"],
            arguments.get("recurring", True),
            arguments.get("durable", True),
        )
        if isinstance(result, str):
            return f"Error: {result}"
        return f"已调度 {result.id}: '{result.cron}' → {result.prompt}"

    def run_list_crons(self, arguments=None) -> str:
        with self.cron_lock:
            jobs = list(self.scheduled_jobs.values())
        if not jobs:
            return "暂无 cron job。可用 schedule_cron 添加。"
        lines = []
        for j in jobs:
            tag = "recurring" if j.recurring else "one-shot"
            dur = "durable" if j.durable else "session"
            lines.append(f"  {j.id}: '{j.cron}' → {j.prompt[:40]} [{tag}, {dur}]")
        return "\n".join(lines)

    def run_cancel_cron(self, arguments) -> str:
        err = require_param(arguments, "job_id")
        if err:
            return err
        return self.cancel_job(arguments["job_id"])

    def register_tools(self, agent_tool):
        agent_tool.register_tool(
            name="schedule_cron",
            description="调度一个 cron 定时任务。cron 为五段式：min hour dom month dow。",
            parameters={
                "type": "object",
                "properties": {
                    "cron": {"type": "string", "description": "五段式 cron 表达式"},
                    "prompt": {"type": "string", "description": "触发时注入给 Agent 的消息"},
                    "recurring": {"type": "boolean", "description": "True=周期性，False=一次性"},
                    "durable": {"type": "boolean", "description": "True=持久化到磁盘"},
                },
                "required": ["cron", "prompt"],
            },
            func=self.run_schedule_cron,
        )
        agent_tool.register_tool(
            name="list_crons",
            description="列出所有已注册的 cron job。",
            parameters={"type": "object", "properties": {}, "required": []},
            func=self.run_list_crons,
        )
        agent_tool.register_tool(
            name="cancel_cron",
            description="按 ID 取消一个 cron job。",
            parameters={
                "type": "object",
                "properties": {"job_id": {"type": "string", "description": "cron job ID"}},
                "required": ["job_id"],
            },
            func=self.run_cancel_cron,
        )