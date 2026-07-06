"""AsyncAgent：Agent 门面类的异步版本。

与同步 Agent 的关系：
- 复用同一套 AgentConfig / AgentPrompt / AgentContextManager / HookManager / AgentTool
- LLM 调用走 AsyncOpenAI（llm_async），工具执行走 aexecute_tool_call（支持 async 工具）
- 主循环 _arun_loop 是 async，可直接 await，天然融入 FastAPI / asyncio 编排

省略同步版里依赖线程的能力（BackgroundTaskManager / cron REPL）——
异步场景用 asyncio.create_task / AgentTeam 编排更自然。
"""

from __future__ import annotations

from typing import Optional

from openai import AsyncOpenAI

from xiaozhi.config import AgentConfig
from xiaozhi.context_manager import AgentContextManager
from xiaozhi.hooks import HookManager
from xiaozhi.llm_async import achat_completion_stream
from xiaozhi.prompt_builder import AgentPrompt
from xiaozhi.statistics import AgentStatistics
from xiaozhi.tool_runner_async import arun_tool_calls
from xiaozhi.tools import AgentTool
from xiaozhi.tracer import tracer

DEFAULT_MAX_TOKENS = 8000


class AsyncAgent:
    """异步 Agent 门面。构造签名与同步 Agent 尽量一致。"""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        *,
        model: str = None,
        api_key: str = None,
        base_url: str = None,
        identity: str = None,
        tools: Optional[list] = None,
        client: Optional[AsyncOpenAI] = None,
        name: str = "agent",
        install_default_hooks: bool = False,
        extra_body: Optional[dict] = None,
        **config_kwargs,
    ):
        if config is None:
            overrides = {k: v for k, v in
                         dict(model=model, api_key=api_key, base_url=base_url,
                              identity=identity).items()
                         if v is not None}
            config = AgentConfig(**overrides, **config_kwargs)
        elif identity is not None:
            config.identity = identity
        self.config = config
        self.config.ensure_dirs()
        self.name = name
        # 透传给 LLM 的 extra_body（如 {"thinking":{"type":"disabled"}} 关推理）
        self.extra_body = extra_body or {}

        self.client = client or AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        self.model = config.model

        # 只在本 agent 明确要求开 trace 时配置全局 tracer；不主动关闭——
        # 避免 build_team 里后构造的 agent（enable_trace=False）把先前开启的 trace 覆盖掉。
        if config.enable_trace:
            tracer.configure(trace_dir=config.trace_dir, enabled=True)

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

        # 异步 Agent 默认不开子 agent 递归（编排交给 AgentTeam）
        self.tool = AgentTool(
            agent_skills=self.skills,
            client=self.client,
            model=self.model,
            enable_subagent=False,
            enable_skills=config.enable_skills,
            max_subagent_depth=config.max_subagent_depth,
        )

        self.context = AgentContextManager(
            client=self.client, model=self.model,
            transcript_dir=config.transcript_dir,
            total=config.context_window, threshold=config.context_threshold,
        )

        self.hooks = HookManager(config.workdir, install_defaults=install_default_hooks)

        for t in (tools or []):
            self.add_tool(t)

        self.messages: list = []

    # ── 工具注册 ──

    def add_tool(self, func_or_spec):
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
        self.hooks.register(event, callback)
        return self

    # ── 系统 prompt ──

    def _sync_system_prompt(self, messages: list) -> list:
        tool_desc = self.tool.render_tools()
        skill_desc = self.skills.render_skills() if self.skills else ""
        relevant = ""  # 异步 memory 检索可在此扩展；PoC 不做
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

    # ── 主循环 ──

    async def _arun_loop(self, messages, max_rounds, print_output=True, context=None):
        # context 可注入：并发 run_once 时用独立 context，避免共享 self.context.cur_window 竞态
        ctx = context or self.context
        # 每一轮都可用工具；轮次耗尽后再单独做一次"强制收尾"调用（禁用工具），
        # 保证：① 不会因禁用工具而让 max_rounds=1 无法调用工具；
        #      ② 无论如何都能拿到最终回答，绝不返回 None。（对齐 MindBridge 的 force_final）
        for _ in range(max_rounds):
            messages = self._sync_system_prompt(messages)
            tracer.snapshot_messages(messages)
            messages = await ctx.aensure_context_limit(messages)

            llm_response, tool_calls, finish_reason = await achat_completion_stream(
                client=self.client,
                messages=messages,
                tools=self.tool.get_openai_tools(),
                model=self.model,
                statistics=self.statistics,
                max_tokens=DEFAULT_MAX_TOKENS,
                print_output=print_output,
                extra_body=self.extra_body,
            )

            if tool_calls:
                messages = ctx.append_message(messages, message={
                    "role": "assistant",
                    "content": llm_response or None,
                    "tool_calls": tool_calls,
                })
                messages = await arun_tool_calls(
                    tool_calls,
                    self.tool.aexecute_tool_call,
                    messages,
                    on_pre_hook=lambda tc: self.hooks.trigger("PreToolUse", tc),
                    on_post_hook=lambda tc, r: self.hooks.trigger("PostToolUse", tc, r),
                    verbose=print_output,
                )
                continue

            messages = ctx.append_message(
                messages=messages, role='assistant', content=llm_response)
            return llm_response

        # 轮次耗尽仍在请求工具：强制收尾一次（禁用工具，明确要求直接作答）
        return await self._force_final(messages, ctx, print_output)

    async def _force_final(self, messages, ctx, print_output):
        """禁用工具，让模型基于已有观察直接给出最终回答，避免返回 None。"""
        ctx.append_message(messages, role="user", content=(
            "已达到工具调用上限。请基于以上观察结果直接给出最终回答，"
            "不要再请求调用任何工具。"))
        messages = self._sync_system_prompt(messages)
        messages = await ctx.aensure_context_limit(messages)
        llm_response, _, _ = await achat_completion_stream(
            client=self.client,
            messages=messages,
            tools=None,
            model=self.model,
            statistics=self.statistics,
            max_tokens=DEFAULT_MAX_TOKENS,
            print_output=print_output,
            extra_body=self.extra_body,
        )
        ctx.append_message(messages=messages, role='assistant', content=llm_response)
        return llm_response

    def _new_context(self):
        """为无状态并发调用创建独立的 context 实例（不共享 cur_window）。"""
        return AgentContextManager(
            client=self.client, model=self.model,
            transcript_dir=self.config.transcript_dir,
            total=self.config.context_window, threshold=self.config.context_threshold,
        )

    # ── 对外 API ──

    async def chat(self, query: str, print_output: bool = True) -> Optional[str]:
        """发送一条用户消息，运行到最终回答。"""
        self.hooks.trigger("UserPromptSubmit", query)
        with tracer.span("turn", f"{self.name} · {query[:50]}", detail=query):
            self.context.append_message(
                messages=self.messages, role='user', content=query)
            return await self._arun_loop(self.messages, self.config.max_rounds, print_output)

    async def run_once(self, query: str, print_output: bool = False) -> Optional[str]:
        """无状态单轮：用独立的 messages + 独立 context 跑一次，不污染 self.messages，
        且并发调用同一个 AsyncAgent 实例时互不干扰（各自独立 context.cur_window）。
        供 AgentTeam 并行调度 worker 时使用。"""
        messages: list = []
        ctx = self._new_context()
        ctx.append_message(messages=messages, role='user', content=query)
        with tracer.span("turn", f"{self.name} · {query[:50]}", detail=query):
            return await self._arun_loop(messages, self.config.max_rounds, print_output, context=ctx)

    def reset(self):
        self.messages = []
        self.context.clear_context()