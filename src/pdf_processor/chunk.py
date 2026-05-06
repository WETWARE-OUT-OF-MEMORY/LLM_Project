# 统一的 chunk 数据结构定义。所有 pdf_processor / retriever / rag_core 共用。
from dataclasses import dataclass, asdict, fields
from typing import Optional, List, Dict, Set, Any

# 公开字段集（写盘 + 入库 + 渲染会用到的字段）
# frozenset: 不能被修改的集合
PUBLIC_FIELDS: frozenset = frozenset({
    'text', 'category',
    'section_title', 'breadcrumb',
    'source_doc', 'source_page',
    'text_as_html', 'image_path', 'text_as_latex', 'summary',
})


@dataclass
class Chunk:
    """PDF 切分后的统一 chunk 表示。
    必填:
        text:     embedding 用的可检索文本
        category: 元素类型 (Title / NarrativeText / Table / Image / Formula / CodeSnippet)
    可选公开字段（按 category 选择性填充）:
        section_title  : 当前所属章节标题
        breadcrumb     : 章节路径，如 "第6章 > 6.1 进程"
        source_doc     : 来源 PDF 文件名
        source_page    : 来源页号列表
        text_as_html   : Table 专用，原始 HTML 结构
        image_path     : Image 专用，图片文件路径
        text_as_latex  : Formula 专用，LaTeX 字符串
        summary        : 可选，LLM 生成的摘要
    内部字段（不写盘、不嵌入）:
        last_title_level: 合并算法用的标题层级状态
    """
    # 必填
    text: str
    category: str

    # 公开可选
    section_title: Optional[str] = None
    breadcrumb: Optional[str] = None
    source_doc: Optional[str] = None
    source_page: Optional[List[int]] = None
    text_as_html: Optional[str] = None
    image_path: Optional[str] = None
    text_as_latex: Optional[str] = None
    summary: Optional[str] = None

    # 内部状态（私有，序列化时过滤）
    last_title_level: int = -1

    def to_public_dict(self) -> Dict[str, Any]:
        # 返回只含公开字段的 dict，可直接 json.dump / 入向量库 metadata。
        return {k: v for k, v in asdict(self).items() if k in PUBLIC_FIELDS}

    @classmethod
    def field_names(cls) -> Set[str]:
        # 所有字段名（含私有），方便测试 / schema 校验
        return {f.name for f in fields(cls)}