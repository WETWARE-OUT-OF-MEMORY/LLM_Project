"""线上查询改写模型"""

import os
from typing import Any, Dict

import yaml
from openai import OpenAI

class OnlineRewriteLLM(OpenAI):

    def __init__(self, config_path: str = "D:/Learn/machine_learning/LLM_Project/config/configs.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        rewrite_config: Dict[str, Any] = self.config.get("llm", {}).get("rewrite_online", {})
        self.model = rewrite_config.get("model", "qwen-plus")
        self.base_url = rewrite_config.get(
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

    def generate(self, prompt: Dict[str, str])->str:
        completion = self.client.chat.completions.create(
            # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            model="qwen-plus",
            messages=[
                {"role": "system", "content": prompt["instructions"]},
                {"role": "user", "content": prompt["input"]},
            ]
        )
        return completion.choices[0].message.content