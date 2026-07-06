import json
import re


def extract_json_array(text: str) -> list | None:
    """Extract a JSON array from LLM response text."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None