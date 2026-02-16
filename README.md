your_rag_project/                 # 项目根目录
│
├── data/                        # 数据目录
│   ├── raw_pdfs/               # 存放原始PDF文件（如408教材）
│   └── processed/              # 存放处理后的文本、向量数据库等（.gitignore忽略）
│
├── src/                        # 源代码目录（核心）
│   ├── __init__.py
│   ├── pdf_processor/          # 模块①：PDF处理子项目
│   │   ├── __init__.py
│   │   ├── pdf_loader.py       # PDF读取与文本提取
│   │   ├── text_splitter.py    # 文本分块策略
│   │   └── utils.py            # 相关工具函数
│   │
│   ├── retriever/              # 模块②&④：检索与RAG核心
│   │   ├── __init__.py
│   │   ├── vector_store.py     # 向量数据库构建与检索
│   │   └── rag_core.py         # RAG链的组装
│   │
│   ├── llm_integration/        # 与大模型交互
│   │   ├── __init__.py
│   │   ├── local_llm.py        # 本地模型调用（如Qwen）
│   │   └── prompt_templates.py # 定义各种提示词模板
│   │
│   └── evaluation/             # 模块⑦：评测（关键！）
│       ├── __init__.py
│       ├── benchmark.py        # 构建评测集
│       └── metrics.py          # 计算检索/生成指标
│
├── configs/                    # 配置文件目录
│   └── config.yaml             # 或 config.py，集中管理路径、模型参数等
│
├── outputs/                    # 运行结果输出（.gitignore忽略）
│   ├── evaluation_results/     # 评测结果与报告
│   └── responses/              # 模型回答样例
│
├── tests/                      # 单元测试（可选但推荐）
│   ├── __init__.py
│   └── test_pdf_processor.py
│
├── requirements.txt            # 项目依赖
├── README.md                   # 项目说明
└── main.py                     # 主程序入口