import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml
import os


class LocalLLM:
    """本地LLM模型封装"""

    def __init__(self, config_path: str = "config/configs.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        model_path = self.config['llm']['local_model_path']
        device = self.config['llm']['device']

        print(f"🔄 正在加载模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )

        if device == "cpu":
            self.model = self.model.to(device)

        self.device = device
        print(f"✅ 模型加载完成，使用设备: {device}")

    def generate(self, prompt: str, max_length: int = 512, temperature: float = 0.7) -> str:
        """生成回答"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取生成的部分（去除prompt）
        if prompt in response:
            response = response[len(prompt):].strip()

        return response
