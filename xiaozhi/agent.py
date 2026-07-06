"""Agent 门面类：把组件组装、主循环、错误恢复、REPL 全部收进一个类。

对外 API：
    from xiaozhi import Agent, tool

    agent = Agent(model="gpt-4o", api_key="...")
    reply = agent.chat("你好")        # 单轮（含多轮工具调用）
    agent.repl()                       # 交互式命令行

自定义工具：
    @tool(description="查天气")
    def get_weather(city: str) -> str: ...

    agent = Agent(tools=[get_weather])
"""

from __future__ import annotations

import random
import threading
import time
from typing import Callable, Optional

from openai import (
    OpenAI,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
    APIStatusError,
)

from xiaozhi.config import AgentConfig
from xiaozhi.context_manager import AgentContextManager
from xiaozhi.hooks import HookManager
from xiaozhi.llm import chat_completion_stream
from xiaozhi.prompt_builder import AgentPrompt
from xiaozhi.statistics import AgentStatistics
from xiaozhi.tool_runner import run_tool_calls
from xiaozhi.tools import AgentTool
from xiaozhi.tracer import tracer
from xiaozhi.background import BackgroundTaskManager

# ── 错误恢复参数 ──
DEFAULT_MAX_TOKENS = 8000
ESCALATED_MAX_TOKENS = 64000
MAX_RECOVERY_RETRIES = 3
MAX_NETWORK_RETRIES = 10


def _retry_delay(attempt, retry_after=None):
    """指数退避 + 抖动；若服务端给了 Retry-After 则优先遵循。"""
    if retry_after:
        return retry_after
    base = min(500 * (2 ** attempt), 32000) / 1000
    return base + random.uniform(0, base * 0.25)


def _parse_retry_after(err):
    """从异常的响应头中解析 Retry-After（秒）；无则返回 None。"""
    resp = getattr(err, "response", None)
    if resp is None:
        return None
    val = resp.headers.get("retry-after")
    if not val:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


class _RecoveryState:
    """单轮 loop 的截断/网络恢复状态。"""

    def __init__(self, model, fallback_model):
        self.has_escalated = False
        self.recovery_count = 0
        self.consecutive_529 = 0
        self.current_model = model
        self.fallback_model = fallback_model


class Agent:
    """自研 Agent 框架的门面类。"""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        *,
        model: str = None,
        api_key: str = None,
        base_url: str = None,
        tools: Optional[list] = None,
        client: Optional[OpenAI] = None,
        **config_kwargs,
    ):
        # 组装配置：显式关键字 > config 对象 > 环境变量
        if config is None:
            overrides = {k: v for k, v in
                         dict(model=model, api_key=api_key, base_url=base_url).items()
                         if v is not None}
            config = AgentConfig(**overrides, **config_kwargs)
        self.config = config
        self.config.ensure_dirs()

        # LLM client
        self.client = client or OpenAI(api_key=config.api_key, base_url=config.base_url)
        self.model = config.model

        # tracer
        tracer.configure(trace_dir=config.trace_dir, enabled=config.enable_trace)

        # ── 组件装配（按开关）──
        self.statistics = AgentStatistics()

        self.memory = None
        if config.enable_memory:
            from xiaozhi.components.memory import AgentMemory
            self.memory = AgentMemory(self.client, self.model, config.memory_dir)

        self.skills = None
        if config.enable_skills:
            from xiaozhi.components.skills import AgentSkills
            self.skills = AgentSkills(config.workdir / "skills")

        self.prompt = AgentPrompt(
            memory=self.memory,
            workspace=str(config.workdir),
            identity=config.identity,
        )

        self.tool = AgentTool(
            agent_skills=self.skills,
            client=self.client,
            model=self.model,
            enable_subagent=config.enable_subagent,
            enable_skills=config.enable_skills,
            max_subagent_depth=config.max_subagent_depth,
        )

        self.context = AgentContextManager(
            client=self.client, model=self.model,
            transcript_dir=config.transcript_dir,
            total=config.context_window, threshold=config.context_threshold,
        )

        self.background = BackgroundTaskManager() if config.enable_background else None

        self.tasks = None
        from xiaozhi.components.tasks import AgentTask
        self.tasks = AgentTask(config.tasks_dir)
        self.tasks.register_tools(self.tool)

        self.cron = None
        if config.enable_cron:
            from xiaozhi.components.cron import AgentCron
            self.cron = AgentCron(config.workdir / ".scheduled_tasks.json")
            self.cron.register_tools(self.tool)

        self.worktree = None
        if config.enable_worktree:
            from xiaozhi.components.worktree import AgentWorktree
            self.worktree = AgentWorktree(config.workdir, config.worktrees_dir, self.tasks)
            self.worktree.register_tools(self.tool)

        self.mcp = None
        if config.enable_mcp:
            from xiaozhi.components.mcp import MCPToolManager
            self.mcp = MCPToolManager()
            self.mcp.register_tools(self.tool)

        # hooks
        self.hooks = HookManager(config.workdir, install_defaults=True)

        # 注册用户自定义工具
        for t in (tools or []):
            self.add_tool(t)

        # 会话状态
        self.messages: list = []
        self._lock = threading.Lock()
        self._trace_server = None

    # ── 工具注册 ──

    def add_tool(self, func_or_spec):
        """注册一个 @tool 装饰过的函数，或 (name, description, parameters, func) 直传。"""
        spec = getattr(func_or_spec, "_xiaozhi_tool", None)
        if spec is None:
            raise TypeError("add_tool 需要一个被 @tool 装饰的函数")
        self.tool.register_tool(
            name=spec.name,
            description=spec.description,
            parameters=spec.parameters,
            func=spec.make_adapter(),
        )
        return self

    def register_hook(self, event: str, callback):
        """追加一个自定义 hook。"""
        self.hooks.register(event, callback)
        return self

    # ── 系统 prompt ──

    def _sync_system_prompt(self, messages: list) -> list:
        """把最新的工具/技能/记忆状态刷进 system prompt（messages[0]）。"""
        tool_desc = self.tool.render_tools()
        skill_desc = self.skills.render_skills() if self.skills else ""
        relevant = self.memory.load_memories(messages) if self.memory else ""
        context = self.prompt.update_context(
            tool_desc=str(tool_desc), skill_desc=str(skill_desc),
            relevant_memories=relevant,
        )
        system_prompt = self.prompt.get_system_prompt(context)
        system_msg = {"role": "system", "content": system_prompt}
        if messages and messages[0].get("role") == "system":
            messages[0] = system_msg
        else:
            messages.insert(0, system_msg)
        return messages

    # ── 网络重试 ──

    def _with_retry(self, fn, state, max_retries=MAX_NETWORK_RETRIES):
        for attempt in range(max_retries):
            try:
                return fn()
            except (RateLimitError, APITimeoutError, APIConnectionError,
                    InternalServerError, APIStatusError) as err:
                status = getattr(err, "status_code", None)
                retry_after = _parse_retry_after(err)
                # 529 过载：尝试 fallback 模型
                if status == 529:
                    state.consecutive_529 += 1
                    if state.fallback_model and state.consecutive_529 >= 2:
                        print(f"\033[33m[recovery] 连续过载，切换到 fallback 模型 "
                              f"{state.fallback_model}\033[0m")
                        state.current_model = state.fallback_model
                else:
                    state.consecutive_529 = 0
                if attempt == max_retries - 1:
                    raise
                delay = _retry_delay(attempt, retry_after)
                print(f"\033[33m[recovery] {type(err).__name__} 第 {attempt + 1} 次，"
                      f"{delay:.1f}s 后重试\033[0m")
                time.sleep(delay)
        return None

    # ── 主循环 ──

    def _run_loop(self, messages, max_rounds):
        state = _RecoveryState(self.model, self.config.fallback_model)
        max_tokens = DEFAULT_MAX_TOKENS
        for _ in range(max_rounds):
            # 消费已触发的 cron job → 注入为 user 消息
            if self.cron:
                for job in self.cron.consume_cron_queue():
                    with tracer.span("cron", f"cron 注入 · {job.prompt[:50]}", detail=job.prompt):
                        messages = self.context.append_message(
                            messages=messages, role="user",
                            content=f"[Scheduled] {job.prompt}",
                        )
                    print(f"  \033[35m[inject cron] {job.prompt[:50]}\033[0m")

            messages = self._sync_system_prompt(messages)
            tracer.snapshot_messages(messages)

            print('====================')
            print(f'上下文使用情况：{self.context.cur_window} / {self.context.TOTAL}')
            print('====================')

            # 发送前压缩
            messages = self.context.ensure_context_limit(messages)

            llm_response, tool_calls, finish_reason = self._with_retry(
                lambda: chat_completion_stream(
                    client=self.client,
                    messages=messages,
                    tools=self.tool.get_openai_tools(),
                    model=state.current_model,
                    statistics=self.statistics,
                    max_tokens=max_tokens,
                    extra_body={"thinking": {"type": "disabled"}},
                ),
                state,
            )

            if finish_reason == "length":
                if not state.has_escalated:
                    print(f"\033[33m[recovery] 输出被截断，max_tokens 提升到 "
                          f"{ESCALATED_MAX_TOKENS} 后重试\033[0m")
                    max_tokens = ESCALATED_MAX_TOKENS
                    state.has_escalated = True
                    continue
                messages = self.context.append_message(
                    messages, message={"role": "assistant", "content": llm_response or None})
                if state.recovery_count < MAX_RECOVERY_RETRIES:
                    messages = self.context.append_message(messages, message={"role": "user", "content":
                        "输出达到 token 上限。请直接续写——不要道歉、不要重述，从中断处继续。"})
                    state.recovery_count += 1
                    print(f"\033[33m[recovery] 续写第 {state.recovery_count}/"
                          f"{MAX_RECOVERY_RETRIES} 次\033[0m")
                    continue
                print("\033[31m[recovery] 多次续写后仍被截断，放弃本轮\033[0m")
                return llm_response

            if tool_calls:
                messages = self.context.append_message(messages, message={
                    "role": "assistant",
                    "content": llm_response or None,
                    "tool_calls": tool_calls,
                })
                messages = run_tool_calls(
                    tool_calls,
                    self.tool.execute_tool_call,
                    messages,
                    on_pre_hook=lambda tc: self.hooks.trigger("PreToolUse", tc),
                    on_post_hook=lambda tc, result: self.hooks.trigger("PostToolUse", tc, result),
                    background_manager=self.background,
                    verbose=True,
                )
                if self.background:
                    notifications = self.background.collect_notifications()
                    if notifications:
                        print(f"  \033[32m[inject] {len(notifications)} 条后台任务通知\033[0m")
                        messages = self.context.append_message(
                            messages=messages, role="user",
                            content="\n\n".join(notifications),
                        )
                continue

            messages = self.context.append_message(
                messages=messages, role='assistant', content=llm_response,
            )

            if self.memory:
                self.memory.extract_memories(messages)
                self.memory.consolidate_memories()

            return llm_response
        print("\033[31m[loop] 达到最大轮次上限，未能得出最终回答\033[0m")
        return None

    # ── 对外 API ──

    def chat(self, query: str) -> Optional[str]:
        """发送一条用户消息，运行 agent 直到得出最终回答，返回回答文本。"""
        with self._lock:
            self.hooks.trigger("UserPromptSubmit", query)
            with tracer.span("turn", f"用户轮次 · {query[:50]}", detail=query):
                self.context.append_message(
                    messages=self.messages, role='user', content=query)
                return self._run_loop(self.messages, self.config.max_rounds)

    def reset(self):
        """清空会话历史。"""
        self.messages = []
        self.context.clear_context()

    def start_trace_server(self, port=8777):
        """启动 trace 可视化网页服务（需 enable_trace=True 才有数据）。"""
        from xiaozhi.components.trace_server import start_trace_server
        self._trace_server = start_trace_server(
            port=port, trace_dir=str(self.config.trace_dir))
        return self._trace_server

    def repl(self):
        """交互式命令行。支持 cron 后台调度（若开启）。"""
        if self.config.enable_trace_server:
            self.start_trace_server()

        self.messages = self._sync_system_prompt(self.messages)

        # cron：启动调度线程 + 空闲投递线程
        if self.cron:
            self.cron.start_scheduler()

            def queue_processor_loop():
                while True:
                    time.sleep(0.2)
                    if not self.cron.has_cron_queue():
                        continue
                    if not self._lock.acquire(blocking=False):
                        continue
                    try:
                        if self.cron.has_cron_queue():
                            print("\n  \033[35m[queue processor] 投递定时任务\033[0m")
                            with tracer.span("turn", "定时任务轮次", detail="(cron)"):
                                self._run_loop(self.messages, self.config.max_rounds)
                    finally:
                        self._lock.release()

            threading.Thread(target=queue_processor_loop, daemon=True).start()
            print("  \033[35m[queue processor] started\033[0m")

        print("小智已就绪。输入 /q 或 /exit 退出。")
        while True:
            try:
                query = input('> ')
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if query.lower() in ['/q', '/exit']:
                self.hooks.trigger("Stop", self.messages)
                break
            self.chat(query)