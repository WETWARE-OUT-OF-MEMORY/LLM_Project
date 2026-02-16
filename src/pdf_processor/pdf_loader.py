import os
from typing import List, Optional

import yaml
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json, elements_from_json

import sys


class PDFLoader:
    """加载并提取PDF文本。支持缓存：已加载的 PDF 从缓存读取，未加载则解析并保存。"""

    def __init__(self, config_path: str = "config/configs.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.raw_pdf_dir = self.config['paths']['raw_pdf_dir']
        self.cache_dir = self.config['paths'].get('pdf_cache_dir', os.path.join(
            self.config['paths']['processed_dir']
        ))

    def _cache_path(self, pdf_filename: str) -> str:
        """根据 PDF 文件名生成缓存文件路径"""
        base = os.path.splitext(pdf_filename)[0]
        return os.path.join(self.cache_dir, f"{base}.json")

    def load_and_extract(self, pdf_filename: str, max_pages: Optional[int] = None) -> List:
        """提取单本PDF的每一页文本。先查缓存，未加载则解析整个 PDF 并保存；已加载则跳过解析。max_pages 为整数时仅返回前 N 页元素。"""
        pdf_path = os.path.join(self.raw_pdf_dir, pdf_filename)
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

        cache_path = self._cache_path(pdf_filename)
        os.makedirs(self.cache_dir, exist_ok=True)

        if os.path.exists(cache_path):
            page_elements = elements_from_json(filename=cache_path)
        else:
            load_config = self.config['pdf_loading']['hi_res_loading']
            partition_kwargs = {k: v for k, v in load_config.items()
                               if k not in ('chunking_strategy', 'max_characters', 'combine_text_under_n_chars')}
            page_elements = partition_pdf(pdf_path, **partition_kwargs)
            with open(cache_path, 'w', encoding='utf-8') as f:
                f.write(elements_to_json(page_elements, indent=2))

        if max_pages is not None and max_pages > 0:
            max_page_num = max_pages
            page_elements = [
                el for el in page_elements
                if getattr(getattr(el, 'metadata', None), 'page_number', 1) <= max_page_num
            ]

        return page_elements
