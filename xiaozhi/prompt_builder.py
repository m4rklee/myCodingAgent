"""系统 prompt 组装器。

与原版差异：``workspace`` 与 ``identity`` 通过构造函数注入，不再引用全局 WORKDIR。
"""

import json
from typing import Optional


class AgentPrompt:
    def __init__(self, memory=None, workspace: str = "", identity: str = ""):
        self.memory = memory
        self.workspace = workspace or "(未指定工作区)"
        self.identity = identity or "你是小智，一个智能助手。你可以使用工具、技能等完成用户的任务。"
        self.memory_hint = "请遵循记忆中的用户偏好；当用户说‘记住’或表达清晰偏好时，将其提取为记忆。"
        self._last_context_key: Optional[str] = None
        self._last_prompt: Optional[str] = None

    def assemble_system_prompt(self, context: dict):
        sections = []

        # ## 角色
        sections.append(f"## 角色\n{self.identity}")

        # ## 环境
        sections.append(f"## 环境\n{context.get('workspace', f'工作区: {self.workspace}')}")

        # ## 工具
        if context.get("tools"):
            sections.append(f"## 工具\n下面是你拥有的工具信息：\n{context['tools']}")

        # ## 技能
        if context.get("skills"):
            sections.append(f"## 技能\n下面是你拥有的技能信息：\n{context['skills']}")

        # ## 记忆
        memories = context.get("memories", "")
        if memories:
            sections.append(f"## 记忆\n{self.memory_hint}\n相关记忆：\n{memories}")

        return "\n\n".join(sections)

    def get_system_prompt(self, context):
        key = json.dumps(context, sort_keys=True, ensure_ascii=False, default=str)
        if key == self._last_context_key and self._last_prompt:
            print("  \033[90m[cache hit] system prompt unchanged\033[0m")
            return self._last_prompt
        self._last_context_key = key
        self._last_prompt = self.assemble_system_prompt(context)

        loaded = ["角色", "环境"]
        if context.get("tools"):
            loaded.append("工具")
        if context.get("skills"):
            loaded.append("技能")
        if context.get("memories"):
            loaded.append("记忆")
        print(f"  \033[32m[assembled] sections: {', '.join(loaded)}\033[0m")
        return self._last_prompt

    def update_context(self, tool_desc="", skill_desc="", relevant_memories=""):
        """从真实状态派生 context：工具/技能渲染结果、记忆索引、相关记忆、工作区。"""
        index = self.memory.read_memory_index() if self.memory else ""
        memories_parts = []
        if index:
            memories_parts.append(f"可用记忆索引：\n{index}")
        if relevant_memories:
            memories_parts.append(relevant_memories)
        return {
            "tools": tool_desc,
            "skills": skill_desc,
            "workspace": f"工作区: {self.workspace}",
            "memories": "\n\n".join(memories_parts),
        }