"""Agent 记忆系统：文件持久化的记忆条目 + LLM 提取/整合/检索。

与原版差异：``memory_dir`` / ``client`` / ``model`` 通过构造函数注入，不再引用全局。
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from xiaozhi.json_utils import extract_json_array
from xiaozhi.message_utils import extract_message_text

MEMORY_TYPES = ["user", "feedback", "project", "reference"]
CONSOLIDATE_THRESHOLD = 10


@dataclass
class MemoryItem:
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class AgentMemory:
    """Agent 记忆系统：工作上下文、情景经验、语义知识、个性化记忆。"""

    def __init__(self, client, model: str, memory_dir: Path):
        self.working_context: List[MemoryItem] = []
        self.episodic_experience: List[MemoryItem] = []
        self.semantic_knowledge: List[MemoryItem] = []
        self.personalized_memory: List[MemoryItem] = []
        self.client = client
        self.model = model
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_index = self.memory_dir / "MEMORY.md"

    def add_working_context(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        self.working_context.append(
            MemoryItem(content=content, metadata=metadata or {})
        )

    @staticmethod
    def extract_text(content):
        """Backward-compatible alias for extract_message_text."""
        return extract_message_text(content)

    def _parse_formatter(self, text: str):
        if not text.startswith("---"):
            return {}, text
        parts = text.split("---", 2)
        if len(parts) < 3:
            return {}, text
        meta = {}
        for line in parts[1].strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip().strip('"').strip("'")
        return meta, parts[2].strip()

    def _rebuild_index(self):
        lines = []
        for f in sorted(self.memory_dir.glob("*.md")):
            if f.name == "MEMORY.md":
                continue
            raw = f.read_text()
            meta, body = self._parse_formatter(raw)
            name = meta.get("name", f.stem)
            desc = meta.get("description", body.split("\n")[0][:80])
            # 写绝对路径，模型可直接用 read_file 读取
            lines.append(f"- [{name}]({f.resolve()}) - {desc}")
        self.memory_index.write_text("\n".join(lines) + "\n" if lines else "")

    def read_memory_index(self):
        if not self.memory_index.exists():
            return ""
        text = self.memory_index.read_text().strip()
        return text if text else ""

    def memory_read(self, filename):
        path = self.memory_dir / filename
        if not path.exists():
            return None
        return path.read_text()

    def memory_search(self):
        pass

    def memory_list(self):
        result = []
        for f in sorted(self.memory_dir.glob("*.md")):
            if f.name == "MEMORY.md":
                continue
            raw = f.read_text()
            meta, body = self._parse_formatter(raw)
            result.append({
                "filename": f.name,
                "name": meta.get("name", f.stem),
                "description": meta.get("description", ""),
                "type": meta.get("type", "user"),
                "body": body,
            })
        return result

    def memory_write(self, name, mem_type, description, body):
        memory_name = name.lower().replace(" ", "-")
        filepath = self.memory_dir / f"{memory_name}.md"
        filepath.write_text(
            f"---\nname: {name}\ndescription: {description}\ntype: {mem_type}\n---\n\n{body}\n"
        )
        self._rebuild_index()
        return filepath

    def select_relevant_memories(self, messages, max_items=5):
        files = self.memory_list()
        if not files:
            return []

        recent_texts = []
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = extract_message_text(msg.get("content", ""))
                if content:
                    recent_texts.append(content)
                if len(recent_texts) >= 3:
                    break
        recent = " ".join(reversed(recent_texts))[:2000]

        if not recent.strip():
            return []

        catalog_lines = []
        for i, f in enumerate(files):
            catalog_lines.append(f"{i}: {f['name']} - {f['description']}")
        catalog = "\n".join(catalog_lines)

        prompt = (
            "Given the recent conversation and the memory catalog below, "
            "select the indices of memories that are clearly relevant. "
            "Return ONLY a JSON array of integers, e.g. [0, 3]. "
            "If none are relevant, return [].\n\n"
            f"Recent conversation:\n{recent}\n\n"
            f"Memory catalog:\n{catalog}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
            )
            text = (response.choices[0].message.content or "").strip()
            indices = extract_json_array(text)
            if indices is not None:
                selected = []
                for idx in indices:
                    if isinstance(idx, int) and 0 <= idx < len(files):
                        selected.append(files[idx]["filename"])
                        if len(selected) >= max_items:
                            break
                return selected
        except Exception:
            pass

        keywords = [w.lower() for w in recent.split() if len(w) > 3]
        selected = []
        for f in files:
            text = (f["name"] + " " + f["description"]).lower()
            if any(kw in text for kw in keywords):
                selected.append(f["filename"])
                if len(selected) >= max_items:
                    break
        return selected

    def load_memories(self, messages: list):
        selected_files = self.select_relevant_memories(messages)
        if not selected_files:
            return ""

        parts = ["##相关记忆"]
        for filename in selected_files:
            content = self.memory_read(filename)
            if content:
                parts.append(content)
        return "\n\n".join(parts)

    def extract_memories(self, messages):
        dialogue_parts = []
        for msg in messages[-10:]:
            role = msg.get("role", "?")
            content = extract_message_text(msg.get("content", ""))
            if content.strip():
                dialogue_parts.append(f"{role}: {content}")
        dialogue = "\n".join(dialogue_parts)

        if not dialogue.strip():
            return

        existing = self.memory_list()
        existing_desc = "\n".join(f"- {m['name']}: {m['description']}" for m in existing) if existing else "(none)"

        prompt = (
            "从对话中提取用户偏好、约束或者项目事实\n"
            "返回格式为JSON列表。每一项包含字段：{name, type, description, body}\n"
            "- name: 短标识符，用'-'连接（如 'user-preference-tabs'）"
            "- type: 类型，取值为 'user'（用户偏好）, 'feedback'（指引）, 'project'（项目事实）或'reference'（外部指向）\n"
            "- description: 一行总结，用于索引查找\n"
            "- body: Markdown格式的全部细节描述\n"
            "如果没有新的记忆或已经被现有记忆涵盖，返回[]\n\n"
            f"现有记忆：\n{existing_desc}\n\n"
            f"对话：\n{dialogue[:4000]}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": prompt}], max_tokens=800
            )
            text = (response.choices[0].message.content or "").strip()
            items = extract_json_array(text)
            if not items:
                return
            count = 0
            for mem in items:
                name = mem.get("name", f"memory_{int(time.time())}")
                mem_type = mem.get("type", "user")
                desc = mem.get("description", "")
                body = mem.get("body", "")
                if desc and body:
                    self.memory_write(name, mem_type, desc, body)
                    count += 1
            if count:
                print(f"\n\033[33m[Memory: extracted {count} new memories]\033[0m")
        except Exception:
            pass

    def consolidate_memories(self):
        files = self.memory_list()
        if len(files) < CONSOLIDATE_THRESHOLD:
            return

        catalog = "\n\n".join(
            f"## {f['filename']}\nname: {f['name']}\ndescription: {f['description']}\n{f['body']}"
            for f in files
        )

        prompt = (
            "整合以下记忆文件。规则：\n"
            "1. 合并重复文件\n"
            "2. 去除过时/相悖的记忆\n"
            "3. 保持记忆总数在30以下\n"
            "4. 优先保留重要的用户偏好\n"
            "返回一个JSON列表。每一项格式：{name, type, description, body}.\n\n"
            f"{catalog[:16000]}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": prompt}], max_tokens=3000
            )
            text = (response.choices[0].message.content or "").strip()
            items = extract_json_array(text)
            if not items:
                return

            for f in self.memory_dir.glob("*.md"):
                if f.name != "MEMORY.md":
                    f.unlink()

            for mem in items:
                name = mem.get("name", f"memory_{int(time.time())}")
                mem_type = mem.get("type", "user")
                desc = mem.get("description", "")
                body = mem.get("body", "")
                if desc and body:
                    self.memory_write(name, mem_type, desc, body)

            print(f"\n\033[33m[Memory: consolidated {len(files)} → {len(items)} memories]\033[0m")
        except Exception:
            pass