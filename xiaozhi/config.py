"""小智 —— 轻量 Agent 框架核心配置与公共类型。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass
class AgentConfig:
    """集中配置：LLM 参数、工作目录、能力开关。

    所有路径都从 ``workdir`` 派生，不依赖全局变量。
    创建后通过 ``Agent(config=...)`` 传入。
    """

    # ── LLM ──
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = ""
    fallback_model: Optional[str] = None

    # ── 行为 ──
    workdir: Optional[Path] = None
    max_rounds: int = 5
    identity: str = ""          # 空时使用内置默认

    # ── 能力开关 ──
    enable_memory: bool = True
    enable_subagent: bool = True
    enable_background: bool = True
    enable_cron: bool = False
    enable_worktree: bool = False
    enable_mcp: bool = False
    enable_skills: bool = True
    enable_trace: bool = False          # tracer 记录
    enable_trace_server: bool = False   # 网页可视化

    # ── 行为调优 ──
    max_subagent_depth: int = 2
    context_window: int = 1_000_000
    context_threshold: int = 10_000

    def __post_init__(self):
        if self.api_key is None:
            load_dotenv()
            self.api_key = os.getenv("LLM_API_KEY")
        if self.base_url is None:
            self.base_url = os.getenv("LLM_BASE_URL")
        if not self.model:
            self.model = os.getenv("LLM_MODEL", "")
        if self.fallback_model is None:
            self.fallback_model = os.getenv("LLM_FALLBACK_MODEL")
        if self.workdir is None:
            self.workdir = Path.cwd()
        else:
            self.workdir = Path(self.workdir)
        if not self.identity:
            self.identity = "你是小智，一个智能助手。你可以使用工具、技能等完成用户的任务。"
        self.api_key = self.api_key or ""
        self.base_url = self.base_url or ""

    # ── 派生路径 ──
    @property
    def memory_dir(self) -> Path:
        return self.workdir / ".memory"

    @property
    def tasks_dir(self) -> Path:
        return self.workdir / ".tasks"

    @property
    def worktrees_dir(self) -> Path:
        return self.workdir / ".worktrees"

    @property
    def transcript_dir(self) -> Path:
        return self.workdir / ".transcripts"

    @property
    def trace_dir(self) -> Path:
        return self.workdir / ".trace"

    def ensure_dirs(self):
        """确保所有会用到的目录存在（幂等）。"""
        for d in (self.memory_dir, self.tasks_dir, self.transcript_dir, self.trace_dir):
            d.mkdir(parents=True, exist_ok=True)


DEFAULT_IDENTITY = "你是小智，一个智能助手。你可以使用工具、技能等完成用户的任务。"