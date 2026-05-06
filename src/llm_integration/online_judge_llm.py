"""线上评委模型调用（OpenAI SDK 兼容模式）。"""

import os
from typing import Any, Dict

import yaml
from openai import OpenAI


class OnlineJudgeLLM:
    """评委模型调用封装（兼容千问/OpenAI）。"""

    def __init__(self, config_path: str = "D:\Learn\machine_learning\LLM_Project\config\configs.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        judge_config: Dict[str, Any] = self.config.get("llm", {}).get("judge_online", {})
        self.model = judge_config.get("model", "qwen-plus")
        self.base_url = judge_config.get(
            "base_url",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ).rstrip("/")
        self.api_key = os.getenv("DASHSCOPE_API_KEY")

        if not self.api_key:
            raise ValueError("Missing DASHSCOPE_API_KEY and llm.judge_online.api_key")

        client_kwargs: Dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self.client = OpenAI(**client_kwargs)

    def generate(self, prompt: Dict[str, str], **kwargs) -> str:
        """调用 chat.completions 并返回文本。"""
        completion = self.client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=[
                {"role": "system", "content": prompt["instructions"]},
                {"role": "user", "content": prompt["input"]},
            ],
        )
        return completion.choices[0].message.content