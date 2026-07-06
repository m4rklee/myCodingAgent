"""AgentTeam：多 Agent 编排原语（Lead 分解 → Worker 并行 → 聚合）。

这是把 MindBridge 的 LeadAgent 逻辑泛化成的可复用框架能力：
- Lead：用 LLM 把用户问题分解成 subtasks，并指派给命名 worker
- Fan-out：asyncio.gather 并行执行各 worker（带超时）
- 聚合：单 worker 成功直接返回；多 worker 由 lead 汇总

设计成与具体业务（心理健康）无关的通用编排器：
    team = AgentTeam(
        lead=AsyncAgent(..., identity=LEAD_PROMPT),
        workers={"consultation": agent1, "diagnostic": agent2, ...},
        decompose_prompt=...,   # 可选：自定义分解指令
        summarize_prompt_fn=..., # 可选：自定义汇总 prompt
    )
    result = await team.run("用户问题")
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from xiaozhi.aio import AsyncAgent
from xiaozhi.llm_async import achat_completion_stream
from xiaozhi.tracer import tracer


@dataclass
class Subtask:
    description: str
    assigned_worker: str


@dataclass
class WorkerResult:
    worker: str
    description: str
    content: str
    ok: bool
    error: Optional[str] = None


@dataclass
class TeamResult:
    subtasks: list[Subtask]
    worker_results: list[WorkerResult]
    summary: str
    complex_task: bool
    timeout_occurred: bool = False
    decomposition_error: Optional[str] = None
    raw_decomposition: Optional[str] = None


DEFAULT_DECOMPOSE_INSTRUCTION = (
    "你是多 Agent 团队的 Lead。请把用户问题分解成 1 到 N 个子任务，"
    "每个子任务指派给一个 worker。\n"
    "**尽量少分配**：能用 1 个 worker 解决就不要用 2 个。\n"
    "只能指派给下列 worker，且必须严格输出 JSON（不要 Markdown、不要解释）：\n"
    '格式：{"subtasks":[{"description":"该 worker 要做什么","assigned_worker":"worker名"}]}\n'
)


class AgentTeam:
    def __init__(
        self,
        lead: AsyncAgent,
        workers: dict[str, AsyncAgent],
        *,
        worker_descriptions: Optional[dict[str, str]] = None,
        decompose_instruction: str = DEFAULT_DECOMPOSE_INSTRUCTION,
        summarize_prompt_fn: Optional[Callable[[str, list[WorkerResult]], str]] = None,
        timeout_seconds: float = 60.0,
    ):
        self.lead = lead
        self.workers = workers
        self.worker_descriptions = worker_descriptions or {
            name: (agent.config.identity or name)[:200] for name, agent in workers.items()
        }
        self.decompose_instruction = decompose_instruction
        self.summarize_prompt_fn = summarize_prompt_fn
        self.timeout_seconds = timeout_seconds

    # ── 分解 ──

    def _decompose_messages(self, user_input: str) -> list[dict]:
        worker_catalog = "\n".join(
            f"- {name}: {desc}" for name, desc in self.worker_descriptions.items()
        )
        return [
            {"role": "system", "content": self.decompose_instruction},
            {"role": "user", "content":
                f"可用 worker：\n{worker_catalog}\n\n用户问题：\n{user_input}"},
        ]

    async def decompose(self, user_input: str) -> tuple[list[Subtask], Optional[str], Optional[str]]:
        """返回 (subtasks, raw, error)。解析失败时回退为单 worker。"""
        messages = self._decompose_messages(user_input)
        try:
            raw, _, _ = await achat_completion_stream(
                client=self.lead.client, messages=messages,
                tools=None, model=self.lead.model, print_output=False,
                extra_body=getattr(self.lead, "extra_body", {}) or {},
            )
        except Exception as exc:
            return self._fallback_subtasks(user_input), None, f"decompose_llm_error: {exc}"

        try:
            subtasks = self._parse_subtasks(raw)
            return subtasks, raw, None
        except Exception as exc:
            return self._fallback_subtasks(user_input), raw, f"decompose_parse_error: {exc}"

    def _parse_subtasks(self, raw: str) -> list[Subtask]:
        payload = self._extract_json_object(raw)
        items = payload.get("subtasks")
        if not isinstance(items, list) or not items:
            raise ValueError("subtasks 必须为非空列表")
        subtasks: list[Subtask] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            desc = str(item.get("description") or "").strip()
            raw_worker = str(item.get("assigned_worker") or item.get("assigned_agent") or "").strip()
            worker = self._resolve_worker(raw_worker)
            if desc and worker:
                subtasks.append(Subtask(desc, worker))
        if not subtasks:
            raise ValueError("没有有效子任务（worker 名不匹配）")
        return subtasks

    def _resolve_worker(self, name: str) -> str | None:
        """把 LLM 给出的 worker 名归一化到实际 workers 的 key。

        兼容常见命名差异：例如 prompt 教模型输出 "research_agent"，
        而 workers 的 key 是 "research"。依次尝试：
        1. 精确匹配
        2. 去掉 _agent/-agent/agent 等后缀再匹配
        3. 双向子串包含匹配（research ⊂ research_agent）
        """
        if not name:
            return None
        if name in self.workers:
            return name
        lowered = name.lower()
        # 去后缀
        stripped = lowered
        for suffix in ("_agent", "-agent", " agent", "agent"):
            if stripped.endswith(suffix):
                stripped = stripped[: -len(suffix)].strip("_- ")
                break
        for key in self.workers:
            if key.lower() == stripped:
                return key
        # 子串包含（任一方向）
        for key in self.workers:
            kl = key.lower()
            if kl and (kl in lowered or lowered in kl):
                return key
        return None

    @staticmethod
    def _extract_json_object(raw: str) -> dict:
        """从可能夹杂解释性文字的 LLM 输出中提取 JSON 对象。

        先试最外层 {..}；失败则用括号配对扫描出每个平衡的顶层对象，
        逐个尝试解析，返回第一个含 "subtasks" 的对象（否则第一个可解析对象）。
        """
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("分解结果未包含 JSON")
        # 快路径：整段就是一个 JSON
        try:
            obj = json.loads(raw[start:end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        # 慢路径：括号配对扫描出所有平衡对象
        candidates = []
        depth, obj_start = 0, -1
        for i in range(start, len(raw)):
            c = raw[i]
            if c == "{":
                if depth == 0:
                    obj_start = i
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0 and obj_start >= 0:
                    try:
                        parsed = json.loads(raw[obj_start:i + 1])
                        if isinstance(parsed, dict):
                            candidates.append(parsed)
                    except json.JSONDecodeError:
                        pass
                    obj_start = -1
        for cand in candidates:
            if "subtasks" in cand:
                return cand
        if candidates:
            return candidates[0]
        raise ValueError("分解结果 JSON 解析失败")

    def _fallback_subtasks(self, user_input: str) -> list[Subtask]:
        # 回退：指派给第一个 worker
        if not self.workers:
            raise ValueError("AgentTeam 没有配置任何 worker，无法执行任务")
        first = next(iter(self.workers))
        return [Subtask(f"处理用户问题：{user_input}", first)]

    # ── fan-out ──

    async def _run_worker(self, subtask: Subtask, user_input: str) -> WorkerResult:
        agent = self.workers[subtask.assigned_worker]
        query = f"子任务：{subtask.description}\n\n用户问题：{user_input}"
        try:
            content = await asyncio.wait_for(
                agent.run_once(query, print_output=False),
                timeout=self.timeout_seconds,
            )
            return WorkerResult(subtask.assigned_worker, subtask.description,
                                content or "", ok=bool(content), error=None)
        except asyncio.TimeoutError:
            return WorkerResult(subtask.assigned_worker, subtask.description,
                                "", ok=False, error="timeout")
        except Exception as exc:
            return WorkerResult(subtask.assigned_worker, subtask.description,
                                "", ok=False, error=str(exc))

    # ── 聚合 ──

    def _default_summary_prompt(self, user_input: str, results: list[WorkerResult]) -> str:
        contributions = "\n\n".join(
            f"**{r.worker}** ({'ok' if r.ok else 'failed'}):\n{r.content or r.error or '无结果'}"
            for r in results
        )
        return (
            "你是多 Agent 团队的 Lead，负责汇总各 worker 的分析结果，生成一个全面、连贯的最终答案。\n\n"
            f"用户问题：{user_input}\n\n各 worker 贡献：\n{contributions}\n\n"
            "请整合以上所有分析，输出结构清晰的最终回答。"
        )

    async def summarize(self, user_input: str, results: list[WorkerResult]) -> str:
        prompt_fn = self.summarize_prompt_fn or self._default_summary_prompt
        prompt = prompt_fn(user_input, results)
        try:
            summary, _, _ = await achat_completion_stream(
                client=self.lead.client,
                messages=[{"role": "user", "content": prompt}],
                tools=None, model=self.lead.model, print_output=False,
                extra_body=getattr(self.lead, "extra_body", {}) or {},
            )
            return (summary or "").strip() or self._fallback_summary(results)
        except Exception:
            return self._fallback_summary(results)

    def _fallback_summary(self, results: list[WorkerResult]) -> str:
        return "\n\n".join(f"【{r.worker}】\n{r.content or r.error or '无结果'}" for r in results)

    # ── 编排入口 ──

    async def run(self, user_input: str) -> TeamResult:
        with tracer.span("turn", f"team · {user_input[:50]}", detail=user_input):
            subtasks, raw, decomp_error = await self.decompose(user_input)

            results = await asyncio.gather(
                *(self._run_worker(st, user_input) for st in subtasks)
            )
            results = list(results)
            timeout_occurred = any(r.error == "timeout" for r in results)

            # 单 worker 成功 → 直接返回，省一次 LLM 汇总
            if len(results) == 1 and results[0].ok and results[0].content.strip():
                summary = results[0].content
            else:
                summary = await self.summarize(user_input, results)

            return TeamResult(
                subtasks=subtasks,
                worker_results=results,
                summary=summary,
                complex_task=len(subtasks) > 1,
                timeout_occurred=timeout_occurred,
                decomposition_error=decomp_error,
                raw_decomposition=raw,
            )