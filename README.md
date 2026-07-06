# 小智 (xiaozhi)

一个自研的、可嵌入的轻量 **Agent 框架**（OpenAI 兼容）。由 AgentLearn 项目重构而来——把原本"能跑的脚本"改造成"能被别的项目 import 的框架"。

## 特性

- **简洁门面**：一个 `Agent` 类聚合所有能力，`agent.chat("...")` 即可用
- **零全局状态**：所有配置、路径、client 都通过 `AgentConfig` 注入，可在同一进程跑多个隔离的 Agent
- **装饰器工具**：`@tool` 把普通带类型注解的函数自动转成工具，无需手写 JSON Schema
- **能力可插拔**（全部可选开关）：
  - 工具调用 + 子 Agent 递归 + 后台任务
  - 记忆系统（自动提取/整合/检索）
  - 多层上下文压缩
  - Cron 定时调度
  - git worktree 隔离
  - MCP 工具接入
  - Skills 加载
  - 调用链 Trace 可视化（网页）

## 安装

```bash
cd xiaozhi-framework
pip install -e .
```

## 配置

通过环境变量或 `.env`：

```bash
LLM_API_KEY=sk-...
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o
LLM_FALLBACK_MODEL=gpt-4o-mini   # 可选：529 过载时切换
```

也可以在代码里显式传入（优先级高于环境变量）。

## 快速开始

### 最小用法

```python
from xiaozhi import Agent

agent = Agent()                       # 从环境变量读取配置
print(agent.chat("你好，介绍下你自己"))
```

### 自定义工具

```python
from xiaozhi import Agent, tool

@tool(description="查询城市天气")
def get_weather(city: str) -> str:
    return f"{city} 今天晴，26℃"

agent = Agent(model="gpt-4o", tools=[get_weather])
print(agent.chat("北京天气怎么样？"))
```

`@tool` 会从函数签名自动推断参数 JSON Schema（`str/int/float/bool/list/dict`、
`Optional`、默认值→非必填）。

### 开启高级能力

```python
from xiaozhi import Agent, AgentConfig

config = AgentConfig(
    model="gpt-4o",
    enable_cron=True,          # 定时任务
    enable_worktree=True,      # git worktree 隔离
    enable_mcp=True,           # MCP 工具
    enable_trace=True,         # 记录调用链
    enable_trace_server=True,  # 网页可视化 (http://127.0.0.1:8777)
)
agent = Agent(config=config)
agent.repl()                   # 交互式命令行
```

### 命令行

```bash
xiaozhi --model gpt-4o --cron --trace-server
```

## API 速览

| 方法 | 说明 |
|------|------|
| `Agent(config=..., tools=[...])` | 构造。也支持 `Agent(model=, api_key=, base_url=)` 直传 |
| `agent.chat(query) -> str` | 发送一条消息，运行到最终回答 |
| `agent.add_tool(func)` | 追加一个 `@tool` 函数 |
| `agent.register_hook(event, cb)` | 追加 hook（`PreToolUse`/`PostToolUse`/`UserPromptSubmit`/`Stop`）|
| `agent.reset()` | 清空会话历史 |
| `agent.repl()` | 交互式命令行（`/q` 退出）|
| `agent.start_trace_server(port)` | 启动 trace 网页 |

## 配置项 (`AgentConfig`)

| 字段 | 默认 | 说明 |
|------|------|------|
| `model` / `api_key` / `base_url` / `fallback_model` | 读环境变量 | LLM 连接 |
| `workdir` | `Path.cwd()` | 工作目录，所有 `.memory/.tasks/...` 都在此下 |
| `max_rounds` | 5 | 单轮对话最大工具调用轮次 |
| `identity` | 内置 | 系统人设 |
| `enable_memory` | True | 记忆系统 |
| `enable_subagent` | True | 子 Agent |
| `enable_background` | True | 后台任务 |
| `enable_cron` | False | Cron 定时 |
| `enable_worktree` | False | git worktree |
| `enable_mcp` | False | MCP 工具 |
| `enable_skills` | True | Skills 加载（`workdir/skills/*/SKILL.md`）|
| `enable_trace` | False | 记录调用链到 `.trace/trace.json` |
| `enable_trace_server` | False | REPL 启动时开网页服务 |

## 目录结构

```
xiaozhi/
├── agent.py           # Agent 门面类（组装 + 主循环 + REPL）
├── config.py          # AgentConfig
├── decorators.py      # @tool
├── hooks.py           # HookManager
├── llm.py             # OpenAI 兼容流式调用
├── tool_runner.py     # 工具执行 + 后台派发
├── tools.py           # AgentTool（内置工具 + 子 Agent）
├── context_manager.py # 多层上下文压缩
├── prompt_builder.py  # 系统 prompt 组装
├── background.py      # 后台任务管理
├── tracer.py          # 调用链记录
├── statistics.py      # token 统计
└── components/        # 可选能力
    ├── memory.py
    ├── skills.py
    ├── tasks.py
    ├── cron.py
    ├── worktree.py
    ├── mcp.py
    └── trace_server.py
```

## 与原 AgentLearn 的关系

本框架是对 `core/` + `harness/` 的**去全局化重构**，逻辑等价但：
- `loop()` 从依赖模块级全局变量的函数 → `Agent._run_loop()` 实例方法
- `WORKDIR = Path.cwd()` 硬编码 → `AgentConfig.workdir` 注入
- import 时的副作用（`load_dotenv`/`mkdir`/hook 自动注册）→ 收敛到 `Agent.__init__`

原项目代码保持不变，仍可独立运行。
