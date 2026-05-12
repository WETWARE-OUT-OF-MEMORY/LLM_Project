import re
import os
import json

from io import BytesIO
from types import SimpleNamespace
from typing import Dict, List, Optional, Set, Tuple

import yaml
from google.protobuf.internal.wire_format import INT32_MAX
from pypdf import PdfReader, PdfWriter
from unstructured.documents.elements import NarrativeText, Element
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json
from dataclasses import replace

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

    def can_combine(self, prev:Chunk, cur:Element) -> bool:
        # 判断prev和cur能否合并
        # 长度限制
        if prev.text is None or cur.text is None:
            return False
        # meta1 = len(prev.text) + len(cur.text) <= 1000
        # 双标题限制: 不都是标题 或者 cur标题比prev标题更深
        meta2 = (prev.category != 'Title' or cur.category != 'Title') \
                 or prev.last_title_level < self.get_title_level(cur.text)
        # 排除正文+标题
        meta3 = not (prev.category == 'NarrativeText' and cur.category == 'Title')
        # SOLO_FIELDS不参与合并
        meta4 = prev.category not in self.SOLO_FIELDS and cur.category not in self.SOLO_FIELDS
        # return meta1 and meta2 and meta3 and meta4
        return meta2 and meta3 and meta4

    def _make_chunk(self, el, title_stack: list) -> Chunk:
        # 从 unstructured Element + 当前 title_stack 构造一个新 Chunk

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
            caption = self.image_describer.describe(_image_path, _section_title, _near_by_text, False)
            el.text = caption or ''

        _section_title = title_stack[-1][1] if title_stack else None
        _breadcrumb = " > ".join(t[1] for t in title_stack[:-1]) if len(title_stack) > 1 else None
        # _text_as_html = el.text_as_html if el.category == 'Table' else None
        # el与metadata为text_as_html兜底
        if el.category == 'Table':
            _text_as_html = getattr(el, 'text_as_html', None) or getattr(el.metadata, 'text_as_html', None)
        else:
            _text_as_html = None
        return Chunk(
            text=el.text or '',
            category=el.category,
            section_title=_section_title,
            breadcrumb=_breadcrumb,
            last_title_level=self.get_title_level(el.text or ''),
            text_as_html=_text_as_html,
            image_path=getattr(el.metadata, 'image_path', None),
        )

    def text_chunk_split(self, chunk:Chunk)->List[Chunk]:
        chunk_size = self.config.get('pdf_splitting',{}).get('chunk_size',1000)
        chunk_overlap = self.config.get('pdf_splitting',{}).get('chunk_overlap',200)
        separators = ["\n\n", "\n", "。", "；", "，", " ", ""]
        piece_list = []
        chunk_list = []

        raw_text = chunk.text
        start = 0
        raw_text_len = len(raw_text)
        while start < raw_text_len:
            end = min(start + chunk_size, raw_text_len)
            window = raw_text[start:end]
            # 尝试在 window 内从后往前找一个合适的分隔符，尽量在语义边界处断开
            split_pos = -1
            for sep in separators[:-1]:  # 最后一个 "" 不参与查找
                idx = window.rfind(sep)
                # 保证分隔点不要离 start 太近，避免产生极短片段
                if idx != -1 and idx > chunk_size * 0.3:
                    split_pos = start + idx + len(sep)
                    break
            if split_pos == -1:
                # 没找到合适分隔符，就在 chunk_size 处硬切
                split_pos = end
            sub_str = raw_text[start:split_pos].strip()
            if sub_str:
                piece_list.append(sub_str)

            # 计算下一个窗口的起点，保留一定重叠
            if split_pos >= raw_text_len:
                break
            start = max(split_pos - chunk_overlap, 0)

        for piece in piece_list:
            chunk_list.append(replace(chunk,text=piece))

        return chunk_list

    def combine_chunks(self, element_list:list[Element])->list:

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
            # 空白表格静默跳过
            if el.category == 'Table' and \
                    (el.text is None or el.text == '') and \
                    (el.text_as_html is None or el.text_as_html == '') and \
                    (el.metadata.text_as_html is None or el.metadata.text_as_html == ''):
                continue

            if el.category == 'Image':
                cnt += 1
                cnt %= CACHE_UPDATE_INTERVAL
                if cnt == 0:
                    self.image_describer.flush()

            if nxt_item is None:
                # nxt_item->Chunk : nxt_item.item
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

        # 对过长纯文本类Chunk按照max_chunk、chunk_overlap进行进一步切分
        splitted_chunks = []
        for chunk in all_chunks:
            if chunk.category in self.TEXT_FIELDS:
                chunks = self.text_chunk_split(chunk)
                for c in chunks:
                    splitted_chunks.append(c)
            splitted_chunks.append(chunk)

        # 转为dict格式，允许JSON序列化
        all_chunks = [c.to_public_dict() for c in splitted_chunks]

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
    pdf_list = ['2025王道操作系统.pdf','2025王道数据结构.pdf','2025王道计算机组成原理.pdf','2025王道计算机网络.pdf']
    for pdf_name in pdf_list:
        chunks = loader.extract_from_pdf(pdf_name)
        min_text_size = INT32_MAX
        max_text_size = -1
        for chunk in chunks:
            if chunk['category'] in loader.TEXT_FIELDS:
                min_text_size = min(min_text_size, len(chunk['text']))
                max_text_size = max(max_text_size, len(chunk['text']))
        print(min_text_size, max_text_size)