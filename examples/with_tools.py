"""自定义工具示例。"""

from xiaozhi import Agent, tool


@tool(description="获取城市的当前天气")
def get_weather(city: str) -> str:
    """返回指定城市的天气信息。"""
    weathers = {"北京": "晴，26℃", "上海": "多云，22℃", "广州": "小雨，28℃"}
    return weathers.get(city, f"{city} 未知")


@tool(description="计算两个数的和")
def add(a: float, b: float) -> str:
    return f"{a} + {b} = {a + b}"


agent = Agent(
    model="gpt-4o",
    tools=[get_weather, add],
)

print(agent.chat("北京天气怎样？"))
print(agent.chat("123 + 456 等于多少？"))