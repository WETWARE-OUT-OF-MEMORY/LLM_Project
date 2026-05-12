import sys
import os
import json
import yaml
import random
import importlib.util
from glob import glob
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pdf_processor.pdf_loader import PDFLoader
from src.pdf_processor.text_splitter import TextSplitter
from src.retriever.vector_store import VectorStore
from src.retriever.rag_core import RAGCore


class RAGSystem:
    """RAG系统主控制器"""

    def __init__(self):
        with open('config/configs.yaml', 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.rag_core = None

    def check_pdf_processed(self) -> bool:
        """检查PDF是否已经处理"""
        raw_files = {
            os.path.splitext(f)[0]
            for f in os.listdir(self.config['paths']['raw_pdf_dir'])
            if f.endswith(".pdf")
        }

        processed_files = {
            os.path.splitext(f)[0]
            for f in os.listdir(self.config['paths']['processed_dir'])
            if f.endswith(".json")
        }

        missing = raw_files - processed_files

        if missing:
            print("缺少处理文件:", missing)
            return False

        return True
    def check_vector_db_built(self) -> bool:
        """检查向量数据库是否已构建"""
        vector_db_path = self.config['paths']['vector_db_path']
        if not os.path.exists(vector_db_path):
            return False
        
        # 检查关键数据文件是否存在
        required_files = [
            os.path.join(vector_db_path, 'prefixed_faiss_index.pkl'),
            os.path.join(vector_db_path, 'texts.json'),
            os.path.join(vector_db_path, 'prefixed_texts.json'),
            os.path.join(vector_db_path, 'metadatas.json')
        ]
        
        # 所有关键文件都必须存在
        return all(os.path.exists(f) for f in required_files)

    def process_pdfs(self):
        """读取并处理PDF文件"""
        # if self.check_pdf_processed():
        #     print("✅ PDF文件已处理，跳过此步骤")
        #     print(f"   缓存位置: {self.config['paths']['processed_dir']}")
        #     return

        print("\n" + "="*50)
        print("📖 开始读取和处理PDF文件")
        print("="*50 + "\n")

        raw_pdf_dir = self.config['paths']['raw_pdf_dir']
        pdf_files = [f for f in os.listdir(raw_pdf_dir) if f.endswith('.pdf')]

        if not pdf_files:
            print("❌ 未找到PDF文件")
            return

        print(f"找到 {len(pdf_files)} 个PDF文件:")
        for i, file in enumerate(pdf_files, 1):
            print(f"  {i}. {file}")

        loader = PDFLoader()
        # splitter = TextSplitter(self.config)
        processed_path = self.config['paths']['processed_dir']
        os.makedirs(processed_path, exist_ok=True)

        # file: /data/RAW路径下的pdf文件路径
        for file in pdf_files:
            print(f"\n处理: {file}")
            # pages = loader.load_and_extract(file)
            # chunks = splitter.split_documents(pages)
            chunks = loader.extract_from_pdf(file)

            chunks_path = os.path.join(processed_path, f'{file.split(".")[0]}.json')
            with open(chunks_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)

            print(f"  ✅ 完成 - 文本块: {len(chunks)}")
            print(f"  💾 保存至: {chunks_path}")

        print("\n✅ 所有PDF处理完成！")

    def build_knowledge_base(self):
        """构建向量知识库"""
        # 检查是否已构建
        if self.check_vector_db_built():
            print("\n⚠️  向量数据库已存在")
            print(f"   位置: {self.config['paths']['vector_db_path']}")
            
            # 询问是否重建
            choice = input("\n是否重新构建? (yes/no): ").strip().lower()
            if choice != 'yes':
                print("❌ 已取消构建")
                return
            
            # 清空现有数据库
            print("\n🔄 清空现有数据库...")
            try:
                vector_store = VectorStore()
                vector_store.clear()
            except Exception as e:
                print(f"⚠️  清空失败: {e}，继续构建...")

        if not self.check_pdf_processed():
            print("\n❌ 请先执行步骤1：读取PDF")
            return

        print("\n" + "="*50)
        print("🗄️  开始构建向量知识库")
        print("="*50 + "\n")

        # 读取所有处理后的chunks
        processed_dir = self.config['paths']['processed_dir']
        all_chunks = []
        
        for file in os.listdir(processed_dir):
            if file.endswith('.json'):
                file_path = os.path.join(processed_dir, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                    all_chunks.extend(chunks)
                print(f"  📁 加载: {file} ({len(chunks)} 个文本块)")

        if not all_chunks:
            print("❌ 未找到处理后的文本块")
            return

        print(f"\n总计: {len(all_chunks)} 个文本块")
        print("🔄 正在构建向量数据库...")

        vector_store = VectorStore()
        vector_store.build_from_chunks(all_chunks)

        print(f"✅ 向量数据库构建完成！")
        print(f"   位置: {self.config['paths']['vector_db_path']}")

    def interactive_qa(self):
        """交互式问答"""
        if not self.check_vector_db_built():
            print("❌ 请先执行步骤2：构建知识库")
            return

        print("\n" + "="*50)
        print("💬 RAG交互式问答系统")
        print("="*50)
        print("提示: 输入问题开始问答，输入 'quit' 或 'exit' 退出\n")

        # 初始化RAG系统
        if self.rag_core is None:
            self.rag_core = RAGCore()

        while True:
            try:
                question = input("\n👤 问题: ").strip()
                
                if question.lower() in ['quit', 'exit', '退出']:
                    print("👋 退出问答模式")
                    break

                if not question:
                    continue

                print("\n🤖 正在思考...")
                result = self.rag_core.answer_with_rag(
                    question,
                    self.config['retrieval']['top_k'],
                    self.config['retrieval']['top_m'],
                    self.config['retrieval']['similarity_threshold']
                )
                
                print(f"\n💡 回答:\n{result['answer']}")
                print(f"\n📚 参考了 {len(result['retrieved_docs'])} 个文档片段")

            except KeyboardInterrupt:
                print("\n\n👋 退出问答模式")
                break
            except Exception as e:
                print(f"\n❌ 错误: {str(e)}")

    def compare_rag_vs_non_rag(self):
        """RAG与非RAG对比问答（从文件读取问题）"""
        if not self.check_vector_db_built():
            print("❌ 请先执行步骤2：构建知识库")
            return
        print("\n" + "="*50)
        print("🔬 RAG vs 非RAG 对比测试")
        print("="*50 + "\n")

        # 询问问题文件路径
        questions_file = 'D:/Learn/machine_learning/LLM_Project/data/questions.txt'

        if not os.path.exists(questions_file):
            print(f"❌ 文件不存在: {questions_file}")
            return

        # 读取问题
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]

        if not questions:
            print("❌ 文件中没有问题")
            return

        print(f"✅ 读取到 {len(questions)} 个问题\n")

        # 初始化RAG系统
        if self.rag_core is None:
            self.rag_core = RAGCore()

        results = []
        
        for i, question in enumerate(questions, 1):
            print(f"\n{'='*50}")
            print(f"问题 {i}/{len(questions)}: {question}")
            print('='*50)

            try:
                # RAG回答
                print("🔄 RAG模式回答中...")
                rag_result = self.rag_core.answer_with_rag(
                    question,
                    self.config['retrieval']['top_k'],
                    self.config['retrieval']['top_m'],
                    self.config['retrieval']['similarity_threshold']
                )
                print(f"✅ RAG回答完成")

                # 非RAG回答
                print("🔄 非RAG模式回答中...")
                non_rag_result = self.rag_core.answer_without_rag(question)
                print(f"✅ 非RAG回答完成")

                results.append({
                    "question": question,
                    "A": rag_result['answer'],
                    "B": non_rag_result['answer']
                })

                # 显示对比
                print(f"\n📝 RAG回答:\n{rag_result['answer'][:]}...")
                print(f"\n📝 非RAG回答:\n{non_rag_result['answer'][:]}...")

            except Exception as e:
                print(f"❌ 错误: {str(e)}")
                results.append({
                    "question": question,
                    "A": f"错误: {str(e)}",
                    "B": f"错误: {str(e)}"
                })

        # 保存结果
        output_dir = os.path.join(self.config['paths']['output_dir'], 'evaluation_results', f'topk_{self.config['retrieval']['top_k']}')
        os.makedirs(output_dir, exist_ok=True)
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, "result.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n\n✅ 对比测试完成！")
        print(f"📊 结果已保存至: {output_file}")

    def view_database_stats(self):
        """查看向量数据库统计信息"""
        if not self.check_vector_db_built():
            print("\n❌ 向量数据库尚未构建")
            return

        print("\n" + "="*50)
        print("📊 向量数据库统计信息")
        print("="*50 + "\n")

        try:
            vector_store = VectorStore()
            stats = vector_store.get_stats()

            print(f"📦 总文本块数: {stats['total_chunks']:,}")
            print(f"📐 向量维度: {stats['embedding_dim']}")
            print(f"📁 数据库路径: {stats['db_path']}")
            
            print("\n文件状态:")
            print(f"  - 索引文件: {'✅' if stats['files_exist']['index'] else '❌'}")
            print(f"  - 文本文件: {'✅' if stats['files_exist']['texts'] else '❌'}")
            print(f"  - 元数据文件: {'✅' if stats['files_exist']['metadatas'] else '❌'}")
            
            if 'index_size_mb' in stats:
                total_size = (stats.get('index_size_mb', 0) + 
                            stats.get('texts_size_mb', 0) + 
                            stats.get('metadatas_size_mb', 0))
                print(f"\n存储空间:")
                print(f"  - 索引: {stats.get('index_size_mb', 0):.2f} MB")
                print(f"  - 文本: {stats.get('texts_size_mb', 0):.2f} MB")
                print(f"  - 元数据: {stats.get('metadatas_size_mb', 0):.2f} MB")
                print(f"  - 总计: {total_size:.2f} MB")

        except Exception as e:
            print(f"❌ 获取统计信息失败: {str(e)}")

    def clear_database(self):
        """清空向量数据库"""
        if not self.check_vector_db_built():
            print("\n⚠️  向量数据库已经是空的")
            return

        print("\n" + "="*50)
        print("🗑️  清空向量数据库")
        print("="*50 + "\n")

        # 显示当前统计

        vector_store = VectorStore()
        stats = vector_store.get_stats()
        print(f"当前数据库包含 {stats['total_chunks']} 个文本块")

        if 'index_size_mb' in stats:
            total_size = (stats.get('index_size_mb', 0) +
                        stats.get('texts_size_mb', 0) +
                        stats.get('metadatas_size_mb', 0))
            print(f"总大小: {total_size:.2f} MB")


        # 二次确认
        print("\n⚠️  警告: 此操作将删除所有向量数据！")
        print("   (PDF处理结果不会被删除，可以重新构建)")
        confirm = input("\n确认清空? (yes/no): ").strip().lower()

        if confirm != 'yes':
            print("❌ 已取消清空操作")
            return

        try:
            vector_store = VectorStore()
            if vector_store.clear():
                print("\n✅ 向量数据库已成功清空！")
                print("   您可以重新执行 [2] 构建向量知识库")
            else:
                print("\n❌ 清空操作失败")
        except Exception as e:
            print(f"\n❌ 清空失败: {str(e)}")

    def run_judge_evaluation(self):
        """调用评委模型对A/B回答进行评分"""
        print("\n" + "="*50)
        print("⚖️  评委模型评分")
        print("="*50 + "\n")

        output_dir = os.path.join(self.config['paths']['output_dir'])
        eval_dir = os.path.join(output_dir, 'evaluation_results')
        eval_py_path = os.path.join(eval_dir, 'eval.py')

        if not os.path.exists(eval_py_path):
            print(f"❌ 未找到评测脚本: {eval_py_path}")
            return

        input_file = os.path.join(output_dir, 'evaluation_results', f'topk_{self.config['retrieval']['top_k']}', 'result.json')
        output_file = os.path.join(output_dir, 'evaluation_results', f'topk_{self.config['retrieval']['top_k']}', 'scoring.json')

        try:
            spec = importlib.util.spec_from_file_location("eval_module", eval_py_path)
            if spec is None or spec.loader is None:
                print("❌ 无法加载评测模块")
                return
            eval_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(eval_module)

            result = eval_module.evaluate_file(
                input_json_path=input_file,
                output_json_path=output_file,
            )

            print("\n✅ 评分完成")
            print(f"📄 输出文件: {result['output_json_path']}")
            print(f"📊 统计: {json.dumps(result['summary'], ensure_ascii=False)}")
        except Exception as e:
            print(f"\n❌ 评分失败: {str(e)}")

    def _load_eval_module(self):
        """动态加载 outputs/evaluation_results/eval.py 模块"""
        output_dir = os.path.join(self.config['paths']['output_dir'])
        eval_dir = os.path.join(output_dir, 'evaluation_results')
        eval_py_path = os.path.join(eval_dir, 'eval.py')

        if not os.path.exists(eval_py_path):
            raise FileNotFoundError(f"未找到评测脚本: {eval_py_path}")

        spec = importlib.util.spec_from_file_location("eval_module", eval_py_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("无法加载评测模块")

        eval_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(eval_module)
        return eval_module

    def analyze_specific_topk(self):
        """分析指定 topk 的 result/scoring 结果"""
        print("\n" + "="*50)
        print("📈 分析指定 TopK 结果")
        print("="*50 + "\n")

        topk_input = input("请输入要分析的 topk（如 1/3/5/8）: ").strip()
        if not topk_input.isdigit():
            print("❌ 输入无效，请输入正整数")
            return
        topk = int(topk_input)
        if topk <= 0:
            print("❌ topk 必须大于 0")
            return

        try:
            eval_module = self._load_eval_module()
            output_dir = os.path.join(self.config['paths']['output_dir'])
            eval_dir = os.path.join(output_dir, 'evaluation_results')
            result = eval_module.analyze_topk_result(topk=topk, evaluation_results_dir=eval_dir)

            print("\n✅ 分析完成")
            print(f"📁 目录: {result['paths']['topk_dir']}")
            print(f"📄 result: {result['paths']['result_json']}")
            print(f"📄 scoring: {result['paths']['scoring_json']}")
            print(f"🧪 一致性检查: {result['consistency_check']}")
            print(f"🏆 胜负统计: {result['winner_count']}")
            print(f"📊 胜率(%): {result['winner_rate_percent']}")
            print(f"🧮 分数统计: {result['score_stats']}")
            print(f"⚠️ 错误数: {result['error_count']}")
        except Exception as e:
            print(f"\n❌ 分析失败: {str(e)}")

    def analyze_all_topk(self):
        """分析 outputs/evaluation_results 下全部 topk_* 结果"""
        print("\n" + "="*50)
        print("📈 分析全部 TopK 结果")
        print("="*50 + "\n")

        try:
            eval_module = self._load_eval_module()
            output_dir = os.path.join(self.config['paths']['output_dir'])
            eval_dir = os.path.join(output_dir, 'evaluation_results')
            result = eval_module.analyze_all_topk_results(evaluation_results_dir=eval_dir)

            print("\n✅ 全部分析完成")
            print(f"📂 基础目录: {result['base_dir']}")
            print(f"🔢 检测到 topk: {result['topk_list']}")
            print(f"⭐ 最佳 topk: {result['best_topk']}")
            print("\n各 topk 摘要:")
            for item in result['reports']:
                topk = item.get('topk')
                if item.get('error'):
                    print(f"  - topk_{topk}: ❌ {item['error']}")
                    continue
                winners = item.get('winner_count', {})
                scores = item.get('score_stats', {})
                print(
                    f"  - topk_{topk}: A/B/Tie/Invalid="
                    f"{winners.get('A', 0)}/{winners.get('B', 0)}/"
                    f"{winners.get('Tie', 0)}/{winners.get('Invalid', 0)}, "
                    f"avgA={scores.get('avg_total_score_A')}, "
                    f"avgB={scores.get('avg_total_score_B')}"
                )

            # ---------------- 消融实验对比 ----------------
            # 目标：观察不同 topk 下 A/B 回答表现的变化趋势与相对基线增益
            valid_reports = [x for x in result.get('reports', []) if not x.get('error')]
            if not valid_reports:
                print("\n🧪 消融实验结果: 无可用数据")
                return

            valid_reports.sort(key=lambda x: x.get('topk', 10**9))
            baseline = valid_reports[0]
            base_topk = baseline.get('topk')
            base_rates = baseline.get('winner_rate_percent', {})
            base_scores = baseline.get('score_stats', {})

            def _num(v, default=0.0):
                return float(v) if isinstance(v, (int, float)) else float(default)

            base_a_rate = _num(base_rates.get('A'))
            base_b_rate = _num(base_rates.get('B'))
            base_gap = _num(base_scores.get('avg_score_gap_A_minus_B'))
            base_avg_a = _num(base_scores.get('avg_total_score_A'))
            base_avg_b = _num(base_scores.get('avg_total_score_B'))

            # 同时构造可落盘文本
            comparison_lines = []
            comparison_lines.append("=" * 60)
            comparison_lines.append("TopK A/B 消融实验对比结果")
            comparison_lines.append("=" * 60)
            comparison_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            comparison_lines.append(f"基线: topk_{base_topk}")
            comparison_lines.append("字段说明: A胜率/B胜率(%)、A均分/B均分、Gap=A均分-B均分")
            comparison_lines.append("         Δ表示相对基线(topk_最小)变化")
            comparison_lines.append("-" * 60)

            print("\n" + "-"*50)
            print("🧪 A/B消融实验（不同 TopK）")
            print("-"*50)
            print(f"基线: topk_{base_topk}")
            print("字段说明: A胜率/B胜率(%)、A均分/B均分、Gap=A均分-B均分")
            print("         Δ表示相对基线(topk_最小)变化")
            print("-"*50)

            for item in valid_reports:
                topk = item.get('topk')
                rates = item.get('winner_rate_percent', {})
                scores = item.get('score_stats', {})

                a_rate = _num(rates.get('A'))
                b_rate = _num(rates.get('B'))
                avg_a = _num(scores.get('avg_total_score_A'))
                avg_b = _num(scores.get('avg_total_score_B'))
                gap = _num(scores.get('avg_score_gap_A_minus_B'))

                delta_a_rate = round(a_rate - base_a_rate, 2)
                delta_b_rate = round(b_rate - base_b_rate, 2)
                delta_gap = round(gap - base_gap, 4)
                delta_avg_a = round(avg_a - base_avg_a, 4)
                delta_avg_b = round(avg_b - base_avg_b, 4)

                line = (
                    f"topk_{topk}: "
                    f"A胜率={a_rate:.2f}% (Δ{delta_a_rate:+.2f}), "
                    f"B胜率={b_rate:.2f}% (Δ{delta_b_rate:+.2f}), "
                    f"A均分={avg_a:.4f} (Δ{delta_avg_a:+.4f}), "
                    f"B均分={avg_b:.4f} (Δ{delta_avg_b:+.4f}), "
                    f"Gap={gap:.4f} (Δ{delta_gap:+.4f})"
                )
                print(line)
                comparison_lines.append(line)
            print("-"*50)

            # 保存到 outputs/evaluation_results/topk_comparation.txt
            output_dir = os.path.join(self.config['paths']['output_dir'])
            eval_dir = os.path.join(output_dir, 'evaluation_results')
            os.makedirs(eval_dir, exist_ok=True)
            comparison_path = os.path.join(eval_dir, "topk_comparation.txt")
            with open(comparison_path, "w", encoding="utf-8") as f:
                f.write("\n".join(comparison_lines) + "\n")
            print(f"💾 消融对比结果已保存: {comparison_path}")
        except Exception as e:
            print(f"\n❌ 全部分析失败: {str(e)}")

    def preview_random_chunk(self):
        """随机展示一段已切分文本（非必选功能）"""
        print("\n" + "="*50)
        print("🎲 随机展示切分文本")
        print("="*50 + "\n")

        processed_dir = self.config['paths']['processed_dir']
        if not os.path.exists(processed_dir):
            print(f"❌ 处理目录不存在: {processed_dir}")
            print("   请先执行 [1] 读取PDF文件")
            return

        json_files = [f for f in os.listdir(processed_dir) if f.endswith('.json')]
        if not json_files:
            print("❌ 未找到切分结果文件")
            print("   请先执行 [1] 读取PDF文件")
            return

        all_chunks = []
        for file in json_files:
            file_path = os.path.join(processed_dir, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and (item.get('text') or '').strip():
                            all_chunks.append(item)
            except Exception as e:
                print(f"⚠️  跳过文件 {file}: {e}")

        if not all_chunks:
            print("❌ 切分结果中没有可展示文本")
            return

        sample = random.choice(all_chunks)
        text = (sample.get('text') or '').strip()
        source_doc = sample.get('source_doc', 'unknown')
        source_page = sample.get('source_page', 'unknown')
        section_title = sample.get('section_title')

        print("✅ 随机抽取成功")
        print(f"📄 来源文档: {source_doc}")
        print(f"📑 来源页码: {source_page}")
        if section_title:
            print(f"🧩 章节标题: {section_title}")
        print(f"📏 文本长度: {len(text)}")
        print("\n--- 文本内容开始 ---")
        print(text)
        print("--- 文本内容结束 ---")

    def show_menu(self):
        """显示主菜单"""
        print("\n" + "="*50)
        print("      RAG 知识库管理系统")
        print("="*50)
        print("\n请选择操作:")
        print("  [1] 读取PDF文件")
        print("  [2] 构建向量知识库")
        print("  [3] 开始交互问答")
        print("  [4] RAG与非RAG对比问答")
        print("  [5] 查看数据库统计")
        print("  [6] 清空向量数据库")
        print("  [7] 评委模型评分")
        print("  [8] 分析指定TopK结果")
        print("  [9] 分析全部TopK结果")
        print("  [10] 随机展示一段切分文本")
        print("  [0] 退出系统")
        print("\n状态:")
        print(f"  📖 PDF已处理: {'✅' if self.check_pdf_processed() else '❌'}")
        print(f"  🗄️  知识库已构建: {'✅' if self.check_vector_db_built() else '❌'}")
        print("="*50)

    def run(self):
        """运行主程序"""
        while True:
            try:
                self.show_menu()
                choice = input("\n请输入选项 (0-10): ").strip()

                if choice == '1':
                    self.process_pdfs()
                elif choice == '2':
                    self.build_knowledge_base()
                elif choice == '3':
                    self.interactive_qa()
                elif choice == '4':
                    self.compare_rag_vs_non_rag()
                elif choice == '5':
                    self.view_database_stats()
                elif choice == '6':
                    self.clear_database()
                elif choice == '7':
                    self.run_judge_evaluation()
                elif choice == '8':
                    self.analyze_specific_topk()
                elif choice == '9':
                    self.analyze_all_topk()
                elif choice == '10':
                    self.preview_random_chunk()
                elif choice == '0':
                    print("\n👋 感谢使用，再见！")
                    break
                else:
                    print("\n❌ 无效选项，请输入0-10之间的数字")

            except KeyboardInterrupt:
                print("\n\n👋 程序已中断，再见！")
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {str(e)}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    system = RAGSystem()
    system.run()