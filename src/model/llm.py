from dataclasses import dataclass

@dataclass
class LLMConfig:
    model: str = ""
    api_key: str | None = None
    base_url: str = ""

@dataclass
class QwenConfig(LLMConfig):
    model: str = "qwen3.6-flash"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

@dataclass
class DeepSeekConfig(LLMConfig):
    model: str = "deepseek-v4-flash"
    base_url: str = "https://api.deepseek.cn/v1"
