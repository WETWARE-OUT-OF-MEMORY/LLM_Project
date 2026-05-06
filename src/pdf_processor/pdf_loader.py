import re
import os
import json

from io import BytesIO
from types import SimpleNamespace
from typing import Dict, List, Optional, Set, Tuple

import yaml
from google.protobuf.internal.wire_format import INT32_MAX
from pypdf import PdfReader, PdfWriter
from unstructured.documents.elements import NarrativeText
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json

from src.pdf_processor.chunk import Chunk
from src.pdf_processor.image_describer import ImageDescriber

class PDFLoader:
    """加载并提取PDF文本。每次均重新解析 PDF，并覆盖缓存。"""
    def __init__(self):
        with open('D:/Learn/machine_learning/LLM_Project/config/configs.yaml', 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.cache_path = self.config['paths']['processed_dir']
        self.file_path = self.config['paths']['raw_pdf_dir']
        self.image_describer = ImageDescriber()
        self.DROP_FIELDS = {'Header','Footer','PageNumber','PageBreak','Address','EmailAddress','CheckBox'}
        self.TEXT_FIELDS = {'NarrativeText','Text','UncategorizedText','FigureCaption'}
        self.SOLO_FIELDS = {'Image','Table','Formula','CodeSnippet'}

    def extract_from_page_list(self, pdf_name: str, pages: Optional[List[int]] = None):
        # 个别页面展示, 返回提取的elements列表
        # pdf_name: 文件名如 file1.pdf
        pdf_file = os.path.join(self.file_path, pdf_name)
        cache_basename = pdf_name

        # 不设置chunking_strategy、max_characters和combine_text_under_n_chars
        kwargs = dict(self.config['pdf_loading']['hi_res_loading'])

        if pages:
            reader = PdfReader(pdf_file)
            n = len(reader.pages)
            for p in pages:
                if p < 1 or p > n:
                    raise ValueError(f"页码 {p} 超出范围 [1, {n}]")
            writer = PdfWriter()
            for p in pages:
                writer.add_page(reader.pages[p - 1])
            buf = BytesIO()
            writer.write(buf)
            buf.seek(0)
            element_list = partition_pdf(filename=None, file=buf, **kwargs)
            stem, _ = os.path.splitext(cache_basename)
            cathe_file = os.path.join(
                self.cache_path, f"{stem}_pages_{'-'.join(map(str, pages))}.json"
            )
        else:
            element_list = partition_pdf(pdf_file, **kwargs)
            cathe_file = os.path.join(self.cache_path, cache_basename)

        elements_to_json(element_list, cathe_file)
        return element_list

    def get_title_level(self, text:str)->int:
        # 标题级别: 章节总标题0级，小标题每有一个.多一级
        # 其他特殊标题返回-1，后续改为正文
        # CHAPTER_RE: 章节总标题正则, eg. 第3章
        # TITLE_RE: 小标题正则，eg. 1.1.3
        CHAPTER_RE = re.compile(r"^\s*第\d+章")
        TITLE_RE = re.compile(r"^\s*\d+(?:\.\d+)*")
        SUBTITLE_RE = re.compile(r"^\s*\d+\.")

        text = (text or "").strip()
        if CHAPTER_RE.match(text):
            return 0
        m = TITLE_RE.match(text)
        if m:
            # 只对匹配的字符串段计数
            # eg. m = re.compile(r"^\d+").match("123sgd")
            # m.groups(): ['123', 'sgd']
            # m.group(1): '123'
            # m.group(2): 'sgd'
            return m.group(0).count('.')
        if SUBTITLE_RE.match(text):
            return INT32_MAX
        return -1

    def can_combine(self, prev, cur) -> bool:
        # 判断prev和cur能否合并
        # 长度限制
        meta1 = len(prev.text) + len(cur.text) <= 1000
        # 双标题限制: 不都是标题 或者 cur标题比prev标题更深
        meta2 = ((prev.category != 'Title' or cur.category != 'Title')
                 or prev.last_title_level < self.get_title_level(cur.text))
        # 排除正文+标题
        meta3 = not (prev.category == 'NarrativeText' and cur.category == 'Title')
        # SOLO_FIELDS不参与合并
        meta4 = prev.category not in self.SOLO_FIELDS and cur.category not in self.SOLO_FIELDS
        return meta1 and meta2 and meta3 and meta4

    def _make_chunk(self, el, title_stack: list) -> Chunk:
        # 从 unstructured Element + 当前 title_stack 构造一个新 Chunk
        _section_title = title_stack[-1][1] if title_stack else None
        _breadcrumb = " > ".join(t[1] for t in title_stack[:-1]) if len(title_stack) > 1 else None
        _text_as_html = el.text_as_html if el.category == 'Table' else None
        return Chunk(
            text=el.text or '',
            category=el.category,
            section_title=_section_title,
            breadcrumb=_breadcrumb,
            last_title_level=self.get_title_level(el.text or ''),
            text_as_html=_text_as_html,
            image_path=getattr(el.metadata, 'image_path', None),
        )

    def combine_chunks(self, element_list:list)->list:

        # VLM缓存最大落盘间隔
        CACHE_UPDATE_INTERVAL = 20
        all_chunks = []
        title_stack = []
        nxt_item = None
        # 处理特殊“标题”: 【*****】
        new_element_list = []
        for el in element_list:
            # 过滤丢弃域
            if el.category in self.DROP_FIELDS:
                continue
            # 分隔独立域
            if el.category in self.SOLO_FIELDS:
                new_element_list.append(el)
                continue
            # 处理文字域
            if el.category in self.TEXT_FIELDS:
                el = NarrativeText(text=el.text, metadata=el.metadata)
            level = self.get_title_level(el.text)
            if level == -1 and el.category == 'Title':
                el = NarrativeText(text=el.text, metadata=el.metadata)
            new_element_list.append(el)
        element_list = new_element_list

        cnt = 0
        for el in element_list:
            # el为标题时更新标题栈
            if el.category == 'Title':
                level = self.get_title_level(el.text)
                while len(title_stack) > 0 and title_stack[-1][0] >= level:
                    title_stack.pop()
                title_stack.append((level, el.text.strip()))

            if el.category == 'Image':
                # VLM生成图片的描述性文本
                _image_path = getattr(el.metadata, 'image_path', None)
                _section_title = title_stack[-1][1] if len(title_stack) > 0 else None
                # 暂时不设计near_by_text
                _near_by_text = None
                caption = self.image_describer.describe(_image_path,_section_title,_near_by_text,False)
                el.text = caption or ''
                cnt += 1
                cnt %= CACHE_UPDATE_INTERVAL
                if cnt == 0:
                    self.image_describer.flush()

            if nxt_item is None:
                # 第一个element成chunk
                nxt_item = self._make_chunk(el, title_stack)
                continue

            if self.can_combine(nxt_item, el):
                # new_element为标题时用双换行分隔，为正文时用单换行分隔
                if nxt_item.category == 'Title':
                    new_text = nxt_item.text + '\n\n' + el.text
                else:
                    new_text = nxt_item.text + '\n' + el.text
                # 合并属性分配
                nxt_item.text = new_text
                # 合并后的类型跟随最后合并的element类型
                nxt_item.category = el.category
                # 如果拼接标题则更新最新标题等级
                if el.category == 'Title':
                    nxt_item.last_title_level = self.get_title_level(el.text)
                # section_title、breadcrumb不做动态更新，冻结为最上层
                # nxt_item.section_title = title_stack[-1][1] if len(title_stack) > 0 else None
                # nxt_item.breadcrumb = " > ".join(title_stack[:-2][1]) if len(title_stack) > 1 else None
            else:
                all_chunks.append(nxt_item)
                nxt_item = self._make_chunk(el, title_stack)
        if nxt_item is not None:
            all_chunks.append(nxt_item)
        self.image_describer.flush()

        # 转为dict格式，允许JSON序列化
        all_chunks = [c.to_public_dict() for c in all_chunks]

        return all_chunks


    def extract_from_pdf(self, pdf_name:str, force_reparse=False):
        pdf_root = self.config['paths']['raw_pdf_dir']
        pdf_path = os.path.join(pdf_root, pdf_name)
        stem,_ = os.path.splitext(pdf_name)
        cache_root = self.config['paths']['processed_dir']
        cache_path = os.path.join(cache_root, f"{stem}.json")

        if not force_reparse and os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        # 当前切分方法中目录中的标题号不予识别；正文中的标题独立拆分为一个element；标题之间的文本视作独立正文；
        # 有效标题格式为: eg.3.1、 3.1.2、 1. ， 可能有前导缩进
        # kwargs = self.config['pdf_loading']['hi_res_loading']
        kwargs = dict(self.config['pdf_loading']['hi_res_loading'])
        element_list = partition_pdf(pdf_path, **kwargs)
        chunks = self.combine_chunks(element_list)

        os.makedirs(cache_root, exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            # indent控制json每行行首缩进空格数
            json.dump(chunks, f, ensure_ascii=False, indent=4)

        return chunks



if __name__ == "__main__":
    loader = PDFLoader()
    pdf_name = '2025王道操作系统.pdf'
    specified_pages = [11, 12, 13]              # 1-based 页码列表
    print("=" * 80)
    print(f"📄 PDF: {pdf_name}")
    print(f"📑 目标页码: {specified_pages}")
    print("=" * 80)
    # 1) 解析指定页 → 原始 element 列表
    element_list = loader.extract_from_page_list(pdf_name, pages=specified_pages)
    print(f"\n[Stage 1] partition_pdf 解析出 {len(element_list)} 个原始元素")
    # 1.1) 原始元素简报（可选：看 unstructured 切了哪些块）
    # print("\n--- 原始元素列表 ---")
    # for i, el in enumerate(element_list, 1):
    #     text_preview = (el.text or "").strip().replace("\n", " ")
    #     if len(text_preview) > 60:
    #         text_preview = text_preview[:60] + "..."
    #     print(f"  [{i:>3}] {el.category:<14} | {text_preview}")
    # 2) 合并 → chunks
    chunks = loader.combine_chunks(element_list)
    print(f"\n[Stage 2] combine_chunks 合并后 {len(chunks)} 个 chunks")
    # 3) 展示每个 chunk
    print("\n" + "=" * 80)
    print("                          切分结果展示")
    print("=" * 80)
    for i, chunk in enumerate(chunks, 1):
        print(f"\n┌─ Chunk {i}/{len(chunks)} " + "─" * 60)
        print(f"│ 类型     : {chunk.get('category')}")
        print(f"│ 章节标题 : {chunk.get('section_title')}")
        print(f"│ 面包屑   : {chunk.get('breadcrumb')}")
        print(f"│ 来源文档 : {chunk.get('source_doc')}")
        print(f"│ 文本长度 : {len(chunk.get('text', ''))}")
        print("├─ 文本内容 " + "─" * 64)
        # 给每行加一个左竖线，方便和外面区分
        body = chunk.get('text', '')
        for line in body.splitlines() or [body]:
            print(f"│ {line}")
        print("└" + "─" * 75)
    print(f"\n✅ 共展示 {len(chunks)} 个 chunks")
