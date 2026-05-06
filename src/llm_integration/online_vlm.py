import yaml
import os
import base64
import mimetypes

from typing import Any, Dict
from openai import OpenAI

class OnlineVLM:
    def __init__(self, config_path: str = "D:\Learn\machine_learning\LLM_Project\config\configs.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        vlm_config: Dict[str, Any] = self.config["llm"]["vlm_online"]
        self.model = vlm_config.get("model", "qwen-vl-plus")
        self.base_url = vlm_config.get(
            "base_url",
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ).rstrip("/")
        self.api_key = os.getenv("DASHSCOPE_API_KEY")

        if not self.api_key:
            raise ValueError("Missing DASHSCOPE_API_KEY and llm.judge_online.api_key")

        client_kwargs: Dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self.client = OpenAI(**client_kwargs)

    def describe_image(self, image_path:str, prompt:str, **kwargs):

        mime, _ = mimetypes.guess_type(image_path)
        if not mime:
            mime = "application/octet-stream"  # 无法识别时的兜底
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        data_url = f"data:{mime};base64,{b64}"

        message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            },
        ]

        params = {k:v for k,v in self.config["llm"]["vlm_online"].items() if k in ("model","temperature","max_tokens")}
        params["messages"] = message

        completion = self.client.chat.completions.create(**params)

        choice = completion.choices[0].message
        text = getattr(choice, "content", None) or ""
        return text.strip()