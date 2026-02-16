from typing import List, Dict


class TextSplitter:
    """将长文本切分为适合检索的片段（自实现版本，避免 langchain 依赖问题）"""

    def __init__(self, config):
        self.chunk_size = config['pdf_splitting']['chunk_size']
        self.chunk_overlap = config['pdf_splitting']['chunk_overlap']
        # 按照一定优先级的分隔符进行粗切，再做长度控制
        self.separators = ["\n\n", "\n", "。", "；", "，", " ", ""]

    def _split_text(self, text: str) -> List[str]:
        """简单实现的递归式文本切分"""
        if not text:
            return []

        chunks: List[str] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            window = text[start:end]

            # 尝试在 window 内从后往前找一个合适的分隔符，尽量在语义边界处断开
            split_pos = -1
            for sep in self.separators[:-1]:  # 最后一个 "" 不参与查找
                idx = window.rfind(sep)
                # 保证分隔点不要离 start 太近，避免产生极短片段
                if idx != -1 and idx > self.chunk_size * 0.3:
                    split_pos = start + idx + len(sep)
                    break

            if split_pos == -1:
                # 没找到合适分隔符，就在 chunk_size 处硬切
                split_pos = end

            chunk = text[start:split_pos].strip()
            if chunk:
                chunks.append(chunk)

            # 计算下一个窗口的起点，保留一定重叠
            if split_pos >= text_len:
                break
            start = max(split_pos - self.chunk_overlap, 0)

        return chunks

    def split_documents(self, pages: List) -> List[Dict]:
        """将页面文本切分为块，并保留元数据（如页码）"""
        all_chunks: List[Dict] = []
        skipped_pages: List[Dict] = []  # 记录跳过的页面信息
        
        for page in pages:
            # 从unstructured元素中提取元数据
            metadata = getattr(page, 'metadata', None)
            page_number = getattr(metadata, 'page_number', None) if metadata else None
            filename = getattr(metadata, 'filename', 'unknown') if metadata else 'unknown'
            
            # 安全地提取文本内容
            page_text = None
            try:
                # 优先使用 text 属性
                page_text = getattr(page, 'text', None)
                
                # 如果 text 为 None，尝试转换为字符串
                if page_text is None:
                    str_result = str(page)
                    # 确保 str() 返回的不是 'None' 字符串且不为空
                    if str_result and str_result != 'None':
                        page_text = str_result
            except Exception as e:
                # 捕获任何转换错误
                print(e)
                pass
            
            # 检查是否成功提取到文本
            if not page_text or not page_text.strip():
                skipped_pages.append({
                    'page': page_number,
                    'file': filename
                })
                print(f"⚠️  跳过空页面: 文件 '{filename}', 第 {page_number} 页 (无可提取文本)")
                continue
            
            # 切分文本
            page_chunks = self._split_text(page_text)
            for chunk in page_chunks:
                all_chunks.append({
                    'text': chunk,
                    'source_page': page_number,
                    'source_doc': filename,
                })
        
        # 输出统计信息
        print(f"文本切分完成，共生成 {len(all_chunks)} 个文本块。")
        if skipped_pages:
            page_nums = [str(p['page']) for p in skipped_pages if p['page'] is not None]
            print(f"📊 统计: 跳过了 {len(skipped_pages)} 个空页面 [页码: {', '.join(page_nums)}]")
        
        return all_chunks