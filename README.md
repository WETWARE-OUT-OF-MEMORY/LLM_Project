## 项目结构

LLM_Project/                          # 项目根目录
├── main.py                           # 命令行入口（RAG / 评测流程）
├── requirements.txt
├── version_explaination.txt
├── README.md
│
├── config/                           # 配置文件（注意：目录名为 config，非 configs）
│   └── configs.yaml                  # 模型、路径、API、检索等参数
│
├── src/
│   ├── pdf_processor/                # PDF 与分块
│   │   ├── pdf_loader.py             # PDF 解析与文本提取
│   │   ├── text_splitter.py          # 文本切分
│   │   ├── chunk.py                # 分块数据结构 / 工具
│   │   └── image_describer.py       # 图片理解（VLM 描述，配合图文检索）
│   │
│   ├── retriever/                    # 向量库与 RAG
│   │   ├── vector_store.py          # 向量构建与相似度检索
│   │   └── rag_core.py              # RAG 组装（可选查询改写、Top-K 等）
│   │
│   ├── llm_integration/              # 大模型与提示词
│   │   ├── local_llm.py             # 本地推理模型（Qwen2.5-1.5B）
│   │   ├── online_rewrite_llm.py    # 在线查询改写（Qwen-plus）
│   │   ├── online_judge_llm.py      # 在线评判 / 打分（Qwen-plus）
│   │   ├── online_vlm.py            # 在线视觉语言模型（Qwen-vl-plus）
│   │   └── prompt_templates.py      # 各类 prompt 构建
│   │
│   └── utils/
│       └── chunk_renderer.py        # 分块渲染等辅助
│
└── outputs/                          # 运行输出
    ├── evaluation_results/           # 当前评测输出（如 topk_8、keypoint.json、eval.py）
    └── prev_evaluation_results/      # 历史评测备份
