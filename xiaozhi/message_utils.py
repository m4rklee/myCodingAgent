def extract_message_text(content) -> str:
    """Extract plain text from OpenAI-compatible message content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(getattr(b, "text", "") or (b.get("text", "") if isinstance(b, dict) else ""))
            for b in content
            if (getattr(b, "type", None) == "text"
                or (isinstance(b, dict) and b.get("type") == "text"))
        )
    return str(content)