"""@tool 装饰器：把一个带类型注解的普通函数转成可注册的工具。

用法：
    @tool(description="查询城市天气")
    def get_weather(city: str) -> str:
        return f"{city} 晴"

装饰后函数带有 ``_xiaozhi_tool`` 属性（含 OpenAI 工具 schema 与调用适配器），
Agent 会在初始化时自动注册。也支持不带参数直接 ``@tool``。
"""

import inspect
from typing import Any, Callable, get_args, get_origin, get_type_hints

# Python 类型 → JSON Schema 类型
_PY_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _json_type(annotation) -> str:
    if annotation in _PY_TO_JSON:
        return _PY_TO_JSON[annotation]
    origin = get_origin(annotation)
    if origin in (list, tuple):
        return "array"
    if origin is dict:
        return "object"
    # Optional[X] / Union → 取第一个非 None
    args = [a for a in get_args(annotation) if a is not type(None)]
    if args:
        return _json_type(args[0])
    return "string"


class ToolSpec:
    """封装一个工具的 OpenAI schema 与「dict 参数 → 函数调用」适配器。"""

    def __init__(self, func: Callable, name: str, description: str,
                 parameters: dict | None = None):
        self.func = func
        self.name = name
        self.description = description
        self.parameters = parameters or self._build_parameters(func)

    @staticmethod
    def _build_parameters(func: Callable) -> dict:
        """从函数签名 + 类型注解生成 JSON Schema。"""
        try:
            hints = get_type_hints(func)
        except Exception:
            hints = {}
        sig = inspect.signature(func)
        props, required = {}, []
        for pname, param in sig.parameters.items():
            if pname in ("self", "cls"):
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL,
                              inspect.Parameter.VAR_KEYWORD):
                continue
            ann = hints.get(pname, str)
            props[pname] = {"type": _json_type(ann)}
            if param.default is inspect.Parameter.empty:
                required.append(pname)
        return {"type": "object", "properties": props, "required": required}

    def make_adapter(self) -> Callable[[dict], str]:
        """返回 func(arguments: dict) -> str 形式的适配器。

        自动识别 async 函数并返回 async 适配器，因此同一个 ToolSpec
        能同时用于同步 Agent.execute_tool_call 和异步 aexecute_tool_call。
        """
        sig = inspect.signature(self.func)
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        valid = {p for p in sig.parameters}

        if inspect.iscoroutinefunction(self.func):
            async def async_adapter(arguments: dict) -> str:
                arguments = arguments or {}
                if accepts_kwargs:
                    kwargs = dict(arguments)
                else:
                    kwargs = {k: v for k, v in arguments.items() if k in valid}
                result = await self.func(**kwargs)
                return result if isinstance(result, str) else str(result)
            return async_adapter
        else:
            def adapter(arguments: dict) -> str:
                arguments = arguments or {}
                if accepts_kwargs:
                    kwargs = dict(arguments)
                else:
                    kwargs = {k: v for k, v in arguments.items() if k in valid}
                result = self.func(**kwargs)
                return result if isinstance(result, str) else str(result)
            return adapter


def tool(_func=None, *, name: str = None, description: str = None,
         parameters: dict = None):
    """把函数标记为工具。可 ``@tool`` 或 ``@tool(description=...)``。"""

    def decorator(func: Callable) -> Callable:
        spec = ToolSpec(
            func=func,
            name=name or func.__name__,
            description=description or (func.__doc__ or "").strip() or func.__name__,
            parameters=parameters,
        )
        func._xiaozhi_tool = spec
        return func

    if _func is not None:
        return decorator(_func)
    return decorator