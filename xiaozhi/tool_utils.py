"""工具参数解析与校验工具函数。无外部依赖。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def parse_openai_tool_call(tool_call) -> tuple[str, dict]:
    """Extract function name and arguments from an OpenAI-compatible tool_call."""
    if isinstance(tool_call, dict):
        function = tool_call.get("function") or {}
        name = function.get("name", "")
        raw_arguments = function.get("arguments", {})
    else:
        function = getattr(tool_call, "function", None)
        name = getattr(function, "name", "") if function else ""
        raw_arguments = getattr(function, "arguments", {}) if function else {}

    if isinstance(raw_arguments, dict):
        arguments = raw_arguments
    elif isinstance(raw_arguments, str) and raw_arguments.strip():
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            arguments = {}
    else:
        arguments = {}

    return name, arguments


def parse_tool_arguments(raw_arguments: str) -> tuple[dict[str, Any], str | None]:
    """Parse tool call arguments JSON; return (arguments, error_message)."""
    if not raw_arguments:
        return {}, None
    try:
        return json.loads(raw_arguments), None
    except json.JSONDecodeError as e:
        return {}, f"Error: 工具参数 JSON 解析失败：{e}; raw_arguments={raw_arguments}"


def require_param(arguments: dict, name: str) -> str | None:
    """Return error message if required parameter is missing, else None."""
    if not arguments.get(name):
        return f"Error: 缺少参数 {name}"
    return None


def read_text_file(path_str: str, param_name: str = "file_path") -> str:
    """Read a UTF-8 text file; return content or error message."""
    if not path_str:
        return f"Error: 缺少参数 {param_name}"
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        return f"Error: 文件不存在：{path}"
    if not path.is_file():
        return f"Error: {path} 不是文件"
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"Error: 文件不是 UTF-8 文本文件：{path}"