import hashlib

import yaml
import os
import json

from typing import Dict, List, Callable, Any, Optional

from src.llm_integration.online_vlm import OnlineVLM
from src.llm_integration.prompt_templates import build_image_description_prompt

PROMPT_VERSION = "v1"

# 图片描述业务层
class ImageDescriber:
    def __init__(self, config_path:str ="D:/Learn/machine_learning/LLM_Project/config/configs.yaml"):
        with open(config_path, 'r', encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.vlm = OnlineVLM(config_path)

        # 加载VLM cache
        self.cache_path = self.config["paths"]["vlm_cache"]
        self._cache: Dict[str, str] = self._load_cache()
        # cache脏键
        self._dirty: bool = False

    def _load_cache(self) -> Dict[str, str]:
        # JSON文件损坏时加载空缓存
        if not os.path.exists(self.cache_path):
            return {}
        try:
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError) as e:
            print(f"⚠️ 缓存加载失败: {e}，使用空缓存")
            return {}

    def _save_cache(self):
        parent = os.path.dirname(self.cache_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = self.cache_path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.cache_path)

    def _cache_key(self,image_path: str, section_title: Optional[str], nearby_text: Optional[str]) -> str:
        # 对 (图片字节 + 上下文 + 模型 + prompt 版本) 一起 sha256
        # 生成稳定的 cache key（sha256 hex）
        h = hashlib.sha256()
        # 1.图片二进制
        with open(image_path, 'rb') as f:
            h.update(f.read())
        # 2.上下文（影响 caption 视角）
        # h.update(str)中str需要转utf-8
        h.update((section_title or '').encode('utf-8'))
        h.update((nearby_text or '').encode('utf-8'))
        # 3.模型名（换模型时旧缓存自动失效）
        h.update(self.vlm.model.encode('utf-8'))
        # 4.prompt 版本号（改 prompt 时旧缓存自动失效）
        h.update(PROMPT_VERSION.encode('utf-8'))
        return h.hexdigest()  # 64 字符的 hex string

    def describe(self, image_path: str, section_title: Optional[str] = None, nearby_text: Optional[str] = None,
                 force: bool = False) -> Optional[str]:
        # 对单张图片生成中文描述
        """流程：
            1) 计算 cache key
            2) 缓存命中且 force=False → 返回缓存
            3) 调 VLM；成功写缓存（不立即持久化）
            4) 失败返回 None，由调用方决定降级（丢弃/占位）
        Args:
            image_path:    图片绝对路径
            section_title: 图所在章节（影响 caption 视角）
            nearby_text:   图相邻段落文本（VLM 上下文增强）
            force:         True 时跳过缓存强制重生成
        Returns:
            caption 字符串；失败时返回 None
        """
        if not os.path.exists(image_path):
            # 图片路径不存在时静默跳过
            return None
        cache_key = self._cache_key(image_path, section_title, nearby_text)
        if cache_key in self._cache.keys() and not force:
            return self._cache[cache_key]
        prompt = build_image_description_prompt(section_title, nearby_text)

        try:
            caption = self.vlm.describe_image(image_path, prompt)
            if not caption or not caption.strip():
                return None
            caption = caption.strip()
        except Exception as e:
            print(f"⚠️ VLM 调用失败: {image_path} → {e}")
            return None

        self._cache[cache_key] = caption
        self._dirty = True
        return caption


    def describe_batch(self,items: List[Dict[str, Any]],save_every: int = 10,
                       on_progress: Optional[Callable[[int, int], None]] = None,) -> List[Optional[str]]:
        # 批量处理多张图（PDFLoader 一次性提交一本 PDF 的所有图）
        # Callable[[a,b,...],c]: [a,b,...]为可调用目标(函数、方法、lambda等)的参数列表，c为返回类型
        """Args:
            items:        [{"image_path": ..., "section_title": ..., "nearby_text": ...}, ...]
            save_every:   每处理 N 张主动 flush 一次（防止崩溃丢失进度）
            on_progress:  可选回调 (done, total)
        Returns:
            与 items 同长的 caption 列表（失败位置为 None）
        """
        caption_batch = []
        for cnt, it in enumerate(items):
            caption = self.describe(
                it["image_path"],
                section_title=it.get("section_title"),
                nearby_text=it.get("nearby_text")
            )
            caption_batch.append(caption)
            if on_progress is not None:
                # 回调显示当前进度
                on_progress(cnt + 1, len(items))
            if (cnt + 1) % save_every == 0:
                self.flush()
        self.flush()
        return caption_batch

    def flush(self):
        # 把内存缓存写盘
        """使用场景：
            - 一批图处理完后调一次（避免每张写一次开销）
            - PDFLoader.extract_from_pdf 末尾调
            - 异常退出 / __del__ 兜底
        """
        if not self._dirty:
            return
        self._save_cache()
        self._dirty = False

    def clear_cache(self) -> int:
        """清空内存 + 磁盘缓存。返回清掉的条目数。"""
        items_num = len(self._cache)
        self._cache.clear()
        self._dirty = False
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        return items_num