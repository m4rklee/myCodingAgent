"""MCP 工具管理（OpenAI 兼容）。

1. 连接 MCP server → 发现工具
2. 把 MCP 工具以 OpenAI 格式动态注册进 AgentTool
3. 工具调用：mcp__{server}__{tool} 前缀路由回对应 server 的 handler

此处内置 mock server 作教学演示；真实场景可替换 MCPClient 的传输实现。
"""

import re
from typing import Any, Callable

from xiaozhi.tool_utils import require_param

_DISALLOWED_CHARS = re.compile(r'[^a-zA-Z0-9_-]')


def normalize_mcp_name(name: str) -> str:
    """把非 [a-zA-Z0-9_-] 的字符替换成下划线。"""
    return _DISALLOWED_CHARS.sub('_', name)


class MCPClient:
    """单个 MCP server 的连接：发现工具 + 调用工具（此处用 mock 教学实现）。"""

    def __init__(self, name: str):
        self.name = name
        self.tools: list[dict[str, Any]] = []          # MCP 原生工具定义
        self._handlers: dict[str, Callable[..., str]] = {}

    def register(self, tool_defs: list[dict], handlers: dict[str, Callable[..., str]]):
        """工具发现：记录 server 暴露的工具定义与本地 handler。"""
        self.tools = tool_defs
        self._handlers = handlers

    def call_tool(self, tool_name: str, args: dict) -> str:
        """工具调用：按原生工具名路由到对应 handler。"""
        handler = self._handlers.get(tool_name)
        if not handler:
            return f"MCP error: 未知工具 '{tool_name}'"
        try:
            return handler(**args)
        except Exception as e:
            return f"MCP error: {type(e).__name__}: {e}"


# ── Mock servers（内置演示，替代真实 stdio/sse 传输）──

def _mock_server_docs() -> MCPClient:
    mcp_client = MCPClient("docs")
    mcp_client.register(
        tool_defs=[
            {"name": "search", "description": "搜索文档。(readOnly)",
             "inputSchema": {"type": "object",
                             "properties": {"query": {"type": "string",
                                                      "description": "搜索关键词"}},
                             "required": ["query"]}},
            {"name": "get_version", "description": "获取 API 版本。(readOnly)",
             "inputSchema": {"type": "object", "properties": {}, "required": []}},
        ],
        handlers={
            "search": lambda query: f"[docs] 为 '{query}' 找到 3 条结果",
            "get_version": lambda: "[docs] API v2.1.0",
        })
    return mcp_client


def _mock_server_deploy() -> MCPClient:
    mcp_client = MCPClient("deploy")
    mcp_client.register(
        tool_defs=[
            {"name": "trigger",
             "description": "触发一次部署。(destructive — 真实场景需审批)",
             "inputSchema": {"type": "object",
                             "properties": {"service": {"type": "string",
                                                        "description": "服务名"}},
                             "required": ["service"]}},
            {"name": "status", "description": "查询部署状态。(readOnly)",
             "inputSchema": {"type": "object",
                             "properties": {"service": {"type": "string",
                                                        "description": "服务名"}},
                             "required": ["service"]}},
        ],
        handlers={
            "trigger": lambda service: f"[deploy] 已触发部署：{service}",
            "status": lambda service: f"[deploy] {service}: running (v1.4.2)",
        })
    return mcp_client


MOCK_SERVERS: dict[str, Callable[[], MCPClient]] = {
    "docs": _mock_server_docs,
    "deploy": _mock_server_deploy,
}


class MCPToolManager:
    """管理已连接的 MCP server，并把它们的工具动态注册进 AgentTool。"""

    def __init__(self):
        self._servers: dict[str, MCPClient] = {}
        self._agent_tool = None          # register_tools 时注入的 AgentTool
        self._registered: set[str] = set()   # 已注册进 AgentTool 的前缀工具名

    # ── 连接 / 发现 ──

    def connect(self, name: str) -> str:
        if name in self._servers:
            return f"MCP server '{name}' 已连接"
        factory = MOCK_SERVERS.get(name)
        if not factory:
            available = ", ".join(MOCK_SERVERS.keys())
            return f"未知的 MCP server '{name}'。可用：{available}"

        mcp_client = factory()
        self._servers[name] = mcp_client
        registered = self._register_server_tools(name, mcp_client)
        tool_names = [t["name"] for t in mcp_client.tools]
        print(f"  \033[31m[mcp] connected: {name} → {tool_names}\033[0m")
        return (f"已连接 MCP server '{name}'，发现 {len(mcp_client.tools)} 个工具："
                f"{', '.join(registered)}")

    def disconnect(self, name: str) -> str:
        if name not in self._servers:
            return f"MCP server '{name}' 未连接"
        del self._servers[name]
        return (f"已断开 MCP server '{name}'（已注册的工具在本次会话内仍可见，"
                "重启后消失）")

    def _register_server_tools(self, server_name: str, mcp_client: MCPClient) -> list[str]:
        """把 server 的每个工具以 OpenAI 格式注册进 AgentTool，返回前缀工具名列表。"""
        safe_server = normalize_mcp_name(server_name)
        prefixed_names: list[str] = []
        for tool_def in mcp_client.tools:
            safe_tool = normalize_mcp_name(tool_def["name"])
            prefixed = f"mcp__{safe_server}__{safe_tool}"
            prefixed_names.append(prefixed)
            if self._agent_tool is None or prefixed in self._registered:
                continue

            # 闭包绑定当前 server 与原生工具名
            def _make_func(client: MCPClient, native: str):
                return lambda arguments: client.call_tool(native, arguments or {})

            self._agent_tool.register_tool(
                name=prefixed,
                description=tool_def.get("description", ""),
                parameters=tool_def.get("inputSchema",
                                        {"type": "object", "properties": {}, "required": []}),
                func=_make_func(mcp_client, tool_def["name"]),
            )
            self._registered.add(prefixed)
        return prefixed_names

    # ── 工具入口（供 LLM 调用）──

    def run_connect_mcp(self, arguments: dict) -> str:
        error = require_param(arguments, "name")
        if error:
            return error
        return self.connect(arguments["name"])

    def run_disconnect_mcp(self, arguments: dict) -> str:
        error = require_param(arguments, "name")
        if error:
            return error
        return self.disconnect(arguments["name"])

    def run_list_mcp_servers(self, arguments=None) -> str:
        if not self._servers:
            available = ", ".join(MOCK_SERVERS.keys())
            return f"当前无已连接的 MCP server。可用：{available}"
        lines = []
        for name, client in self._servers.items():
            tool_names = [t["name"] for t in client.tools]
            lines.append(f"  {name}: {', '.join(tool_names)}")
        return "\n".join(lines)

    # ── 集成入口：注册管理型工具 ──

    def register_tools(self, agent_tool):
        """注册 MCP 管理工具，并保存 agent_tool 引用以便动态注册 MCP 工具。"""
        self._agent_tool = agent_tool
        # 已连接的 server（若在注册前就 connect 过）补注册其工具
        for name, client in self._servers.items():
            self._register_server_tools(name, client)

        agent_tool.register_tool(
            name="connect_mcp",
            description="连接一个 MCP server（可选：docs, deploy）并发现其工具。"
                        "发现的工具以 mcp__{server}__{tool} 命名，会在下一轮自动可用。",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string",
                             "description": "MCP server 名称，如 docs 或 deploy"},
                },
                "required": ["name"],
            },
            func=self.run_connect_mcp,
        )
        agent_tool.register_tool(
            name="disconnect_mcp",
            description="断开一个已连接的 MCP server。",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "MCP server 名称"},
                },
                "required": ["name"],
            },
            func=self.run_disconnect_mcp,
        )
        agent_tool.register_tool(
            name="list_mcp_servers",
            description="列出当前已连接的 MCP server 及其工具。",
            parameters={"type": "object", "properties": {}, "required": []},
            func=self.run_list_mcp_servers,
        )