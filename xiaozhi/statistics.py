class AgentStatistics:
    """Token 用量统计，累加每次 LLM 调用的 usage 信息。"""

    def __init__(self):
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        self.cached_tokens: int = 0

    def update_token_usage(self, usage):
        # 用 getattr 兜底：不同 provider 的 usage 字段不一致。
        # prompt_cache_hit_tokens 是 DeepSeek 扩展字段，标准 OpenAI / 通义等没有，
        # 直接访问会 AttributeError 导致整个 LLM 调用崩溃。
        self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
        self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        self.total_tokens += getattr(usage, "total_tokens", 0) or 0
        self.cached_tokens += getattr(usage, "prompt_cache_hit_tokens", 0) or 0

    def display_token_usage(self):
        return self.total_tokens