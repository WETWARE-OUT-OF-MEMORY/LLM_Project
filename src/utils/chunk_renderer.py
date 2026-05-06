# src/utils/chunk_renderer.py
from typing import Dict, Any


def render_chunk_for_llm(chunk: Dict[str, Any]) -> str:
    # 根据 chunk.category 选择最优表示形式喂给 LLM。
    cat = chunk.get('category', '')

    if cat == 'Table':
        html = chunk.get('text_as_html')
        if html:
            ctx = chunk.get('section_title') or ''
            return f"[表格 - {ctx}]\n{html}"
        return chunk.get('text', '')   # HTML 缺失时退回扁平

    if cat == 'Image':
        caption = chunk.get('text', '')
        path = chunk.get('image_path', '')
        return f"[图片描述]\n{caption}\n(原图路径: {path})"

    if cat == 'Formula':
        latex = chunk.get('text_as_latex') or chunk.get('text', '')
        return f"[公式]\n$$ {latex} $$"

    # 默认：标题 / 正文 / 代码 / 其他
    return chunk.get('text', '')


def render_chunks_for_llm(chunks: list, with_breadcrumb: bool = True) -> str:
    # 批量渲染多个召回 chunk，拼成一段上下文。
    blocks = []
    for c in chunks:
        body = render_chunk_for_llm(c)
        if with_breadcrumb and c.get('breadcrumb'):
            body = f"【{c['breadcrumb']}】\n{body}"
        blocks.append(body)
    return "\n\n---\n\n".join(blocks)