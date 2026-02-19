import os
from typing import Dict, List, Optional, Set, Tuple

import yaml
from pypdf import PdfReader
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json

class PDFLoader:
    """加载并提取PDF文本。每次均重新解析 PDF，并覆盖缓存。"""

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

    @staticmethod
    def _normalize_text(text: str) -> str:
        """去除所有空白字符，用于判断页面是否存在有效文本。"""
        return "".join((text or "").split())

    def _get_total_pages(self, pdf_path: str) -> Optional[int]:
        """获取 PDF 真实总页数。失败时返回 None。"""
        try:
            reader = PdfReader(pdf_path)
            return len(reader.pages)
        except Exception as exc:  # pragma: no cover - 环境依赖/损坏文件导致
            print(f"[PDFLoader] 警告：无法读取 PDF 总页数，将按解析结果估算。原因: {exc}")
            return None

    def _build_page_text_map(self, page_elements: List) -> Dict[int, str]:
        """将解析元素按页聚合为文本。"""
        page_text_map: Dict[int, List[str]] = {}
        for el in page_elements:
            page_number = getattr(getattr(el, "metadata", None), "page_number", None)
            if page_number is None:
                page_number = 1
            page_text_map.setdefault(page_number, []).append(getattr(el, "text", "") or "")
        return {p: "\n".join(texts) for p, texts in page_text_map.items()}

    def _detect_blank_pages(self, page_elements: List, total_pages: int) -> List[int]:
        """识别疑似无法识别页：无元素，或元素文本仅空白。"""
        page_text_map = self._build_page_text_map(page_elements)
        blank_pages: List[int] = []
        for page_num in range(1, total_pages + 1):
            merged_text = page_text_map.get(page_num, "")
            if not self._normalize_text(merged_text):
                blank_pages.append(page_num)
        return blank_pages

    def _retry_read_blank_pages(self, pdf_path: str, blank_pages: List[int]) -> Set[int]:
        """使用另一种加载配置重读，返回成功恢复出文本的页号集合。"""
        if not blank_pages:
            return set()

        fallback_config = self.config.get("pdf_loading", {}).get("fast_loading")
        if not fallback_config:
            print("[PDFLoader] 未配置 fast_loading，跳过二次重读。")
            return set()

        partition_kwargs = {
            k: v for k, v in fallback_config.items()
            if k not in ("chunking_strategy", "max_characters", "combine_text_under_n_chars")
        }
        fallback_elements = partition_pdf(pdf_path, **partition_kwargs)
        fallback_page_text_map = self._build_page_text_map(fallback_elements)

        recovered_pages: Set[int] = set()
        for page_num in blank_pages:
            if self._normalize_text(fallback_page_text_map.get(page_num, "")):
                recovered_pages.add(page_num)
        return recovered_pages

    def _normalize_pdf_name(self, pdf_name: str) -> str:
        """标准化 PDF 文件名（自动补全 .pdf 后缀）"""
        name = pdf_name.strip()
        if not name:
            raise ValueError("pdf_name 不能为空")
        if not name.lower().endswith(".pdf"):
            name = f"{name}.pdf"
        return name

    def _resolve_pdf_path_by_name(self, pdf_name: str, pdf_dir: Optional[str] = None) -> Tuple[str, str]:
        """根据目录和文件名解析 PDF 绝对路径，返回 (pdf_path, normalized_name)"""
        normalized_name = self._normalize_pdf_name(pdf_name)
        search_dir = pdf_dir or self.raw_pdf_dir
        pdf_path = os.path.join(search_dir, normalized_name)

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
        return pdf_path, normalized_name

    def _extract_from_pdf_path(self, pdf_path: str, cache_key_filename: str,
                               max_pages: Optional[int] = None) -> List:
        """从 PDF 路径提取页面元素。每次强制重新解析并覆盖缓存。"""
        cache_path = self._cache_path(cache_key_filename)
        os.makedirs(self.cache_dir, exist_ok=True)

        load_config = self.config['pdf_loading']['hi_res_loading']
        partition_kwargs = {k: v for k, v in load_config.items()
                           if k not in ('chunking_strategy', 'max_characters', 'combine_text_under_n_chars')}
        page_elements = partition_pdf(pdf_path, **partition_kwargs)
        with open(cache_path, 'w', encoding='utf-8') as f:
            f.write(elements_to_json(page_elements, indent=2))

        total_pages = self._get_total_pages(pdf_path)
        if total_pages is None:
            total_pages = max(
                [getattr(getattr(el, "metadata", None), "page_number", 1) or 1 for el in page_elements],
                default=1
            )

        blank_pages = self._detect_blank_pages(page_elements, total_pages)
        if blank_pages:
            print(f"[PDFLoader] {cache_key_filename} 疑似无法识别页（真实页号）: {blank_pages}")
            recovered_pages = self._retry_read_blank_pages(pdf_path, blank_pages)
            if recovered_pages:
                recovered_sorted = sorted(recovered_pages)
                print(f"[PDFLoader] {cache_key_filename} 二次重读成功页号: {recovered_sorted}")
            final_unrecognized = [p for p in blank_pages if p not in recovered_pages]
            if final_unrecognized:
                print(f"[PDFLoader] {cache_key_filename} 最终仍无法识别页号: {final_unrecognized}")
            else:
                print(f"[PDFLoader] {cache_key_filename} 所有疑似无法识别页均已在二次重读中恢复。")

        if max_pages is not None and max_pages > 0:
            max_page_num = max_pages
            page_elements = [
                el for el in page_elements
                if getattr(getattr(el, 'metadata', None), 'page_number', 1) <= max_page_num
            ]

        return page_elements

    def load_and_extract(self, pdf_filename: str, max_pages: Optional[int] = None) -> List:
        """提取单本PDF的每一页文本。默认从 raw_pdf_dir 按文件名读取并强制更新缓存。"""
        pdf_path, normalized_name = self._resolve_pdf_path_by_name(pdf_filename, self.raw_pdf_dir)
        return self._extract_from_pdf_path(pdf_path, normalized_name, max_pages)

    def load_and_extract_by_name(self, pdf_name: str, pdf_dir: Optional[str] = None,
                                 max_pages: Optional[int] = None) -> List:
        """按指定目录+文件名加载 PDF 并提取文本（强制更新缓存，不依赖 main 交互流程）。"""
        pdf_path, normalized_name = self._resolve_pdf_path_by_name(pdf_name, pdf_dir)
        return self._extract_from_pdf_path(pdf_path, normalized_name, max_pages)
