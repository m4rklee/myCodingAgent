# 小智 · xiaozhi

> 一个自研的、可嵌入的轻量 **Agent 框架** —— OpenAI 兼容，同步/异步双模态，内建多 Agent 编排。

`xiaozhi` 用最小依赖（`openai` + `pydantic`）实现了一套完整的 Agent 运行时：ReAct 主循环、工具调用、上下文压缩、多 Agent 协作编排，以及记忆 / 技能 / 定时任务 / MCP / 调用链追踪等可插拔组件。目标是**既能一行 `agent.chat()` 上手，也能作为库支撑真实多 Agent 应用**。

---

## ✨ 特性

| | |
|---|---|
| 🎯 **简洁门面** | 一个 `Agent` / `AsyncAgent` 类聚合全部能力，`agent.chat("...")` 即用 |
| ⚡ **同步 + 异步双模态** | 同步 `Agent` 用于脚本/CLI；异步 `AsyncAgent` 无缝接入 FastAPI / asyncio 并发 |
| 🤝 **多 Agent 编排** | 两种范式：`AgentTeam`（Lead 分解 → Worker 并行 → 聚合，fan-out/fan-in）+ `Team`（长期 teammate + jsonl 收件箱消息总线 + 自动认领任务 + 团队协议）|
| 🛠 **装饰器工具** | `@tool` 从函数类型注解自动生成 JSON Schema，无需手写 |
| 🧩 **能力可插拔** | 记忆 / 技能 / 任务图 / cron / git worktree / MCP / 调用链可视化，按需开关 |
| 🧠 **多层上下文压缩** | 裁剪 / 占位 / 大结果落盘 / LLM 摘要四级压缩，长对话不爆 token |
| 🪝 **Hook 机制** | `PreToolUse` / `PostToolUse` / `UserPromptSubmit` / `Stop` 四类事件可挂载 |
| 🧯 **鲁棒性** | 网络重试 + 429 fallback 模型切换；工具调用轮次上限 + 强制收尾防死循环 |
| 💉 **零全局状态** | 配置/路径/client 全部依赖注入，同进程可跑多个隔离 Agent |

---

## 🚀 快速开始

### 安装

```bash
pip install -e .
```

### 配置（环境变量或 `.env`）

```bash
LLM_API_KEY=sk-...
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o
LLM_FALLBACK_MODEL=gpt-4o-mini   # 可选：服务过载(429)时自动切换
```

### 1) 最小用法

```python
from xiaozhi import Agent

agent = Agent()                       # 从环境变量读取配置
print(agent.chat("你好，介绍下你自己"))
```

### 2) 自定义工具

```python
from xiaozhi import Agent, tool

@tool(description="查询城市天气")
def get_weather(city: str) -> str:
    return f"{city} 今天晴，26℃"

agent = Agent(model="gpt-4o", tools=[get_weather])
print(agent.chat("北京天气怎么样？"))
```

`@tool` 自动从函数签名推断参数 Schema（`str/int/float/bool/list/dict`、`Optional`、默认值→非必填），并自动识别 `async` 函数。

### 3) 异步 + 多 Agent 编排

```python
import asyncio
from xiaozhi import AsyncAgent, AgentTeam

lead = AsyncAgent(model="gpt-4o", name="lead")
team = AgentTeam(
    lead=lead,
    workers={
        "researcher": AsyncAgent(model="gpt-4o", identity="你是资料检索专家", tools=[...]),
        "analyst":    AsyncAgent(model="gpt-4o", identity="你是分析专家",   tools=[...]),
    },
)

result = asyncio.run(team.run("帮我调研并分析一下..."))
print(result.summary)          # Lead 汇总的最终回答
print(result.subtasks)         # Lead 如何分派
print(result.worker_results)   # 各 Worker 的独立产出
```

编排流程：**Lead 用 LLM 把问题分解成子任务并指派 → `asyncio.gather` 并行执行 → 单 Worker 直返 / 多 Worker 由 Lead 汇总**。分解失败、Worker 超时/异常均有降级兜底。

### 4) 开启高级组件 & 交互式 REPL

```python
from xiaozhi import Agent, AgentConfig

agent = Agent(config=AgentConfig(
    model="gpt-4o",
    enable_cron=True,          # 定时任务调度
    enable_worktree=True,      # git worktree 隔离
    enable_mcp=True,           # MCP 工具接入
    enable_trace=True,         # 记录调用链
    enable_trace_server=True,  # 调用链网页可视化
))
agent.repl()                   # 交互式命令行（/q 退出）
```

命令行：`xiaozhi --model gpt-4o --cron --trace-server`

---

## 🏗 架构

```
                    ┌─────────────────────────────────┐
   用户输入 ───────▶│   Agent / AsyncAgent  (门面)     │
                    │   ReAct 主循环 + 强制收尾         │
                    └───────┬─────────────────┬────────┘
                            │                 │
              ┌─────────────▼───┐   ┌─────────▼──────────┐
              │ PromptBuilder   │   │ ContextManager     │
              │ 系统 prompt 组装 │   │ 四级上下文压缩      │
              └─────────────────┘   └────────────────────┘
                            │
              ┌─────────────▼───────────────┐      ┌──────────────────┐
              │ AgentTool  工具注册/执行     │◀────▶│ HookManager      │
              │ (@tool / 内置 / MCP / skill) │      │ 权限·日志·拦截    │
              └─────────────┬───────────────┘      └──────────────────┘
                            │
        ┌───────────────────┼───────────────────────────┐
        ▼                   ▼                            ▼
  可插拔组件          LLM (流式)                     Tracer
  memory/skills/     同步 llm / 异步 llm_async       调用链树 → 网页可视化
  tasks/cron/
  worktree/mcp

        多 Agent：AgentTeam = Lead(AsyncAgent) + Workers(AsyncAgent) 并行编排
```

**设计理念**：
- **门面 + 依赖注入**：`Agent` 聚合组件，但所有状态（config/client/路径）从外部注入，无模块级全局 —— 同进程可并存多个隔离 Agent，也便于测试。
- **同步/异步共享一套组件**：`Agent` 与 `AsyncAgent` 复用同一套 config / prompt / context / hooks / 工具注册，仅 LLM 调用与主循环分同步/异步两版。
- **能力即插件**：每个高级能力（记忆/cron/worktree/mcp…）是独立组件，通过 `AgentConfig` 开关注入，核心循环不感知。
- **编排与执行分离**：`AgentTeam` 只负责"谁做什么、怎么并行、怎么聚合"，不关心 Agent 内部如何 ReAct —— 编排原语与业务无关，可复用于任意多 Agent 场景。

---

## 📖 API 速览

| 方法 | 说明 |
|------|------|
| `Agent(config=, tools=)` / `AsyncAgent(...)` | 构造，支持 `model=/api_key=/base_url=` 直传 |
| `agent.chat(query) -> str` | 同步：运行到最终回答 |
| `await aagent.chat(query)` / `run_once(query)` | 异步：`run_once` 无状态单轮（供并发编排）|
| `team.run(query) -> TeamResult` | 多 Agent 编排（分解→并行→聚合）|
| `@tool(description=...)` | 把函数标记为工具（自动 Schema，支持 async）|
| `agent.add_tool(func)` | 追加工具 |
| `agent.register_hook(event, cb)` | 挂载 hook |
| `agent.repl()` | 交互式命令行 |

---

## ⚙️ 配置项（`AgentConfig` 摘选）

| 字段 | 默认 | 说明 |
|------|------|------|
| `model` / `api_key` / `base_url` / `fallback_model` | 读环境变量 | LLM 连接 |
| `workdir` | `Path.cwd()` | 工作目录（`.memory/.tasks/...` 落此）|
| `max_rounds` | 5 | 单轮最大工具调用轮次 |
| `enable_memory` / `enable_subagent` / `enable_background` | True | 记忆 / 子 Agent / 后台任务 |
| `enable_cron` / `enable_worktree` / `enable_mcp` | False | 定时 / worktree / MCP |
| `enable_skills` | True | Skills 加载（`workdir/skills/*/SKILL.md`）|
| `enable_trace` / `enable_trace_server` | False | 调用链记录 / 网页可视化 |

---

## 📂 目录结构

```
xiaozhi/
├── agent.py           # Agent 门面（同步：组装 + ReAct 主循环 + REPL）
├── aio.py             # AsyncAgent（异步门面 + 主循环）
├── orchestration.py   # AgentTeam 多 Agent 编排原语（fan-out/fan-in）
├── config.py          # AgentConfig（依赖注入配置）
├── decorators.py      # @tool 装饰器
├── hooks.py           # HookManager
├── llm.py / llm_async.py         # OpenAI 兼容流式调用（同步/异步）
├── tool_runner.py / tool_runner_async.py   # 工具执行 + 后台派发
├── tools.py           # AgentTool（内置工具 + 子 Agent）
├── context_manager.py # 四级上下文压缩
├── prompt_builder.py  # 系统 prompt 组装
├── background.py      # 后台任务管理
├── tracer.py          # 调用链记录
├── statistics.py      # token 统计
└── components/        # 可插拔能力：memory / team / skills / tasks / cron / worktree / mcp / trace_server
```

## 🧠 记忆系统（参考 A-Mem, NeurIPS 2025）

`components/memory.py` 实现记忆**写入 → 建链进化 → 召回 → 固化**全生命周期：

- **结构化记忆**：记忆按 `user/feedback/project/reference` 四类分类建索引；LLM 抽取 `keywords/context/tags` 结构化属性，检索返回带时间戳的结构化多字段上下文。
- **自进化建链**：新记忆入库时基于向量近邻 + LLM 决策自动与既有记忆建链，并反向更新关联记忆的 `context/tags`，形成可演化的链式记忆网络（缓解传统记忆「只增不改」）。
- **双路召回 + 多跳扩展**：三级规则（类型/关键词/新鲜度）+ LLM 语义双路融合，沿 `links` 做多跳邻居扩展。
- **固化**：记忆超阈值触发 LLM 合并去重、清理过时记忆。
- **文件新鲜度**：sha256 校验，内容未变的文件不重复读入上下文，省 token。

**召回档位**（`MEMORY_MODE` 开关，逐级增强、可回退）：

| 档位 | 检索 | 依赖 |
|------|------|------|
| `lite`（默认） | 三级规则 + LLM 语义双路 | 零第三方依赖 |
| `hybrid` | dense 向量 + BM25 + 规则 三路融合 | sentence-transformers, rank-bm25 |
| `persist` | ChromaDB 持久化向量 + 三路融合 | + chromadb |
| `official` / `official_eval` | A-Mem 官方算法忠实复刻（对标基准） | 同上 |

> 向量档依赖惰性加载：`pip install -e ".[memory]"` 才需要；`lite` 档零依赖即可运行。

## 🤝 多 Agent 协作

框架提供**两种互补的多 Agent 范式**：

**1. `AgentTeam`（orchestration.py）—— fan-out / fan-in**
Lead 用 LLM 把问题分解成 subtasks 指派给命名 worker，`asyncio.gather` 并行执行，最后聚合。适合可一次性并行分解的单个问题。

**2. `Team`（components/team.py）—— 长期 teammate + 消息总线**
Lead 派生后台常驻 teammate，双方通过**基于 jsonl 收件箱的消息总线**异步通信：

- **消息总线**：每个 Agent 一个 `.mailboxes/<name>.jsonl` 收件箱，消费式读取（读完即删）。
- **teammate 生命周期**：WORK（收件箱→LLM→工具循环）→ IDLE（每 5s 轮询收件箱/任务板）→ SHUTDOWN。
- **自动认领**：teammate 空闲时扫描任务板，认领 pending、无 owner、依赖已满足的任务；任务绑定 worktree 时自动切到隔离目录，实现并发改动隔离。
- **团队协议**：`shutdown` / `plan_approval` 请求-响应，靠 `request_id` 关联，Lead 可要求 teammate 先提交计划再执行。

```python
from xiaozhi.components.team import Team

team = Team(client, model, workdir, tasks, worktree)
team.register_tools(lead_agent_tool)          # 给 Lead 注册 spawn/send/check_inbox/协议工具
team.spawn_teammate("frontend", "前端工程师", "实现登录页")
for msg in team.consume_lead_inbox():         # 主循环每轮消费 Lead 收件箱
    ...                                        # teammate 发回的 result / 协议响应自动路由
```

## 🧪 示例

`examples/` 下有三个可运行示例：`minimal.py`（最小）、`with_tools.py`（自定义工具）、`full_features.py`（全能力）。

## 📝 License

MIT
