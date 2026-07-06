"""Skills 管理：发现、显示、装载 SKILL.md。

与原版差异：``skills_dir`` 通过构造函数注入，不再引用全局 WORKDIR。
"""

import re
import json
import yaml
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Skill:
    name: str
    description: str
    path: str
    content: str


class AgentSkills:
    def __init__(self, skills_dir: Path):
        self.skills_dir = Path(skills_dir)
        self.skills: dict[str, Skill] = {}
        self.register_skills()

    def parse_skill_md(self, skill_file: Path):
        text = skill_file.read_text(encoding='utf-8')
        if not text.startswith("---"):
            return {}, text.strip()
        parts = text.split('---', 2)
        if len(parts) < 3:
            return {}, text.strip()
        raw_metadata = parts[1].strip()
        content = parts[2].strip()
        metadata = yaml.safe_load(raw_metadata) or {}
        if not isinstance(metadata, dict):
            raise ValueError(f"{skill_file} 必须是 YAML 格式")

        return metadata, content

    def register_skills(self):
        if not self.skills_dir.is_dir():
            return

        for skill_path in sorted(self.skills_dir.glob("*/SKILL.md")):
            try:
                metadata, content = self.parse_skill_md(skill_path)
            except Exception as e:
                print(f"  \033[31m[skills] 跳过 {skill_path}: 解析失败 {e}\033[0m")
                continue
            skill_name = metadata.get("name")
            skill_description = metadata.get("description", "")
            if not skill_name:
                print(f"  \033[31m[skills] 跳过 {skill_path}: 缺少 name 字段\033[0m")
                continue
            self.skills[skill_name] = Skill(skill_name, skill_description, str(skill_path), content)

    def render_skills(self):
        return [
            {
                'name': skill.name,
                "description": skill.description,
                'path': skill.path
            }
            for skill in self.skills.values()
        ]

    def parse_skill_calls(self, text):
        pattern = r"<skill_call>\s*(.*?)\s*</skill_call>"
        matches = re.findall(pattern, text, flags=re.DOTALL)

        skill_calls = []
        for raw_json in matches:
            try:
                data = json.loads(raw_json)
            except json.JSONDecodeError:
                skill_calls.append({"name": "__parse_error__"})
                continue
            name = data.get('name')

            if not name:
                skill_calls.append({
                    "name": "__parse_error__",
                    "parameters": {
                        "error": "missing skill name",
                        "raw_text": raw_json,
                    },
                })
                continue

            skill_calls.append({"name": name})
        return skill_calls

    def load_skill(self, skill_calls):
        for skill in skill_calls:
            return self.skills[skill['name']].content