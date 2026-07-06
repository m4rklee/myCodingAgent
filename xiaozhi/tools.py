"""Agent 工具管理：内置工具、工具注册、OpenAI 工具调用、子 agent 递归。

与原版差异：
- ``client``/``model`` 通过构造函数注入（供 spawn_subagent 使用），不再 get_client/get_model。
- 子 agent 深度上限由 ``max_subagent_depth`` 配置。
- 可通过 ``enable_subagent=False`` 关闭子 agent 工具。
"""

import threading
from typing import Any, Callable, List, Optional

from xiaozhi.llm import chat_completion_stream
from xiaozhi.tool_runner import run_tool_calls
from xiaozhi.tool_utils import read_text_file, require_param

# 每线程独立的深度计数，避免后台并发串扰
_subagent_depth = threading.local()


def _current_depth() -> int:
    return getattr(_subagent_depth, "value", 0)


class AgentTool:
    """Agent 工具管理：内置工具、工具注册、OpenAI 工具调用。"""

    def __init__(self, agent_skills=None, client=None, model: str = "",
                 enable_subagent: bool = True, enable_skills: bool = True,
                 max_subagent_depth: int = 2):
        self.tools: List[dict[str, Any]] = []
        self.tool_funcs: dict[str, Callable[[dict[str, Any]], str]] = {}
        self.agent_skills = agent_skills
        self.client = client
        self.model = model
        self.max_subagent_depth = max_subagent_depth

        self.register_tool(
            name='read_file',
            description='读取指定路径的文件内容',
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "要读取的文件路径。"
                    }
                },
                "required": ["file_path"],
            },
            func=self.read_file,
        )
        if enable_skills:
            self.register_tool(
                name='load_skill',
                description='装载Skill（按名称或文件路径）',
                parameters={
                    "type": "object",
                    "properties": {
                        "skill_path": {
                            "type": "string",
                            "description": "技能名称或 SKILL.md 文件路径"
                        }
                    },
                    "required": ["skill_path"]
                },
                func=self.load_skill,
            )
        if enable_subagent:
            self.register_tool(
                name='spawn_subagent',
                description='将任务分派给子agent，可以是一些比较耗时的、输出结果比较多的',
                parameters={
                    "type": "object",
                    "properties": {
                        "task_description": {
                            "type": "string",
                            "description": "任务介绍"
                        },
                        "run_in_background": {
                            "type": "boolean",
                            "description": "是否在后台异步执行（耗时/输出多时建议开启），结果完成后通过任务通知返回"
                        }
                    },
                    "required": ["task_description"]
                },
                func=self.spawn_subagent,
            )

    def register_tool(self, name: str, description: str, parameters: dict[str, Any],
                      func: Callable[[dict[str, Any]], str]):
        if name in self.tool_funcs:
            raise ValueError(f"工具已存在：{name}")
        self.tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            }
        })
        self.tool_funcs[name] = func

    def get_openai_tools(self):
        return self.tools

    def render_tools(self):
        return [tool["function"] for tool in self.tools]

    def execute_tool_call(self, name: str, arguments: dict[str, Any]):
        if name not in self.tool_funcs:
            return f"Error: 未知工具：{name}"
        try:
            return self.tool_funcs[name](arguments)
        except Exception as e:
            return f"Error: 工具执行失败：{type(e).__name__}: {e}"

    async def aexecute_tool_call(self, name: str, arguments: dict[str, Any]):
        """异步执行工具。若工具函数返回 awaitable 则 await 之，
        因此同一个 AgentTool 既能跑同步工具也能跑 async 工具。"""
        import inspect

        if name not in self.tool_funcs:
            return f"Error: 未知工具：{name}"
        try:
            result = self.tool_funcs[name](arguments)
            if inspect.isawaitable(result):
                result = await result
            return result
        except Exception as e:
            return f"Error: 工具执行失败：{type(e).__name__}: {e}"

    def read_file(self, arguments):
        return read_text_file(arguments.get("file_path"), param_name="file_path")

    def load_skill(self, arguments):
        skill_path = arguments.get("skill_path")
        error = require_param(arguments, "skill_path")
        if error:
            return error
        if self.agent_skills and skill_path in self.agent_skills.skills:
            return self.agent_skills.skills[skill_path].content
        return read_text_file(skill_path, param_name="skill_path")

    def spawn_subagent(self, arguments):
        description = arguments.get("task_description")
        error = require_param(arguments, "task_description")
        if error:
            return error

        # 深度保护：超过上限直接拒绝，迫使当前层用现有工具自行完成
        depth = _current_depth()
        if depth >= self.max_subagent_depth:
            return (f"Error: 子agent 嵌套已达上限（{self.max_subagent_depth} 层），已终止派发。"
                    "请在当前层用现有工具直接完成任务，不要再调用 spawn_subagent。")

        from xiaozhi.tracer import tracer

        sub_tools = self.tools
        messages = [{"role": "user", "content": description}]

        _subagent_depth.value = depth + 1
        try:
            with tracer.span("subagent", f"子agent · {str(description)[:60]}",
                             detail=str(description)[:300]):
                for _ in range(30):
                    llm_response, tool_calls, _ = chat_completion_stream(
                        client=self.client,
                        messages=messages,
                        tools=sub_tools,
                        model=self.model,
                        statistics=None,
                        print_output=True,
                    )

                    if not tool_calls:
                        return llm_response

                    messages.append({
                        "role": "assistant",
                        "content": llm_response or None,
                        "tool_calls": tool_calls,
                    })

                    messages = run_tool_calls(
                        tool_calls,
                        self.execute_tool_call,
                        messages,
                    )

                return "Error: 子agent达到最大轮数限制"
        finally:
            _subagent_depth.value = depth