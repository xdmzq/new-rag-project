# SOAR / 开源 RAG

这个仓库是一个多知识库 RAG 项目，包含：

- 一个 Flask Web 应用
- 一个可直接调用的命令行检索入口
- 一套从 Excel / 文档资产构建 RAG 知识库的批处理流水线

当前仓库里已经包含多套已生成的知识库目录，例如 `SpaceRAG_v2`、`半导体_RAG`、`军工_RAG`、`ITSoftwareRAG`、`SmartManufacturingRAG_v2` 等。

## 目录概览

```text
.
├── README.md
└── soar/
    ├── .env
    ├── requirements.txt
    ├── query_spacerag_v2.py
    ├── spacerag_v2_search.py
    ├── rag_eval_cases_template.jsonl
    ├── scripts/
    │   ├── run_batch_pipeline.py
    │   ├── build_rag_from_excel.py
    │   ├── extract_document_texts.py
    │   ├── extract_document_keywords.py
    │   └── convert_excel_tables_to_markdown.py
    ├── main/
    │   ├── app.py
    │   ├── api.py
    │   └── semicDatabase.sqlite3
    ├── SpaceRAG_v2/
    ├── 半导体_RAG/
    ├── 军工_RAG/
    └── ...
```

## 环境准备

建议使用 Python 3.11+。

安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r soar/requirements.txt
```

## 配置说明

项目默认从 `soar/.env` 读取配置。常见变量包括：

```env
FLASK_SECRET_KEY=
DATABASE_URL=
DB_MODE=sqlite
SQLITE_PATH=main/semicDatabase.sqlite3

EMBEDDING_PROVIDER=
DASHSCOPE_API_KEY=
DASHSCOPE_CHAT_MODEL=
DASHSCOPE_EMBEDDING_MODEL=
DASHSCOPE_EMBEDDING_DIM=
SILICONFLOW_API_KEY=
```

说明：

- `DB_MODE` 支持 `sqlite`、`mysql`、`auto`
- 如果不配置 MySQL，Web 应用默认会落到 `soar/main/semicDatabase.sqlite3`
- 构建知识库和部分检索能力依赖 `SILICONFLOW_API_KEY`
- 不要把真实密钥提交到仓库

## 快速开始

### 1. 启动 Web 应用

从仓库根目录执行：

```bash
python3 soar/main/app.py
```

默认监听：

```text
http://127.0.0.1:5002
```

补充说明：

- 启动时会自动创建数据库表
- 应用会自动扫描已生成的知识库目录，只展示资源完整的类别
- 健康检查接口为 `GET /healthz`

### 2. 运行命令行检索

默认查询 `soar/SpaceRAG_v2`：

```bash
cd soar
python3 query_spacerag_v2.py "国内商业火箭发射市场规模和增长趋势如何？"
```

输出 JSON：

```bash
cd soar
python3 query_spacerag_v2.py "低轨卫星互联网的发展驱动因素有哪些？" --json
```

切换到其他知识库目录：

```bash
cd soar
python3 query_spacerag_v2.py "EDA行业有哪些核心企业？" --rag-dir 半导体_RAG --json
```

查看帮助：

```bash
cd soar
python3 query_spacerag_v2.py --help
```

## 重建知识库

### 1. 准备输入数据

流水线默认从 `source_data` 读取原始数据。如果你的数据不在仓库根目录下，请显式传入 `--data-root`。

示例：

```bash
python3 soar/scripts/run_batch_pipeline.py \
  --data-root /path/to/source_data \
  --output-dir /path/to/SpaceRAG_v2 \
  --dry-run
```

### 2. 批处理流水线

流水线包含 5 个阶段：

1. `normalization.py`
2. `convert_excel_tables_to_markdown.py`
3. `extract_document_texts.py`
4. `extract_document_keywords.py`
5. `build_rag_from_excel.py`

执行完整流程：

```bash
python3 soar/scripts/run_batch_pipeline.py \
  --data-root /path/to/source_data \
  --output-dir /path/to/SpaceRAG_v2 \
  --api-key "$SILICONFLOW_API_KEY"
```

常用参数：

- `--dry-run`：只打印每个阶段将要执行的命令
- `--bootstrap-text-first`：先自动补建标注工作簿，再继续后续阶段
- `--skip-normalization` / `--skip-tables` / `--skip-texts` / `--skip-keywords` / `--skip-rag`
- `--skip-embeddings`：只生成结构化 JSONL，不生成向量
- `--disable-faiss`：不生成 FAISS 索引
- `--clear-output`：构建前清空输出目录

查看帮助：

```bash
python3 soar/scripts/run_batch_pipeline.py --help
```

### 3. 关于输出目录

注意这里有一个容易混淆的点：

- 构建脚本默认输出目录是 `SpaceRAG`
- 命令行查询脚本默认读取目录是 `SpaceRAG_v2`

如果你希望新构建的数据能直接被 `query_spacerag_v2.py` 使用，建议：

- 要么把 `--output-dir` 显式指定为你想使用的目录，例如 `soar/SpaceRAG_v2`
- 要么查询时通过 `--rag-dir` 指向你实际生成的目录

## 已接入的知识库类别

代码里已定义的 V2 类别包括：

- `space`
- `semiconductor`
- `military_special`
- `biotech_health`
- `it_software`
- `internet`
- `communication_electronics_hardware`
- `intelligent_manufacturing`
- `new_materials`
- `energy_environmental`
- `consumer_modern_services`

Web 端只会启用那些资源文件完整的类别。一个 V2 知识库目录至少应包含：

- `docs.jsonl`
- `pages.jsonl`
- `chunks.jsonl`
- `tables.jsonl`
- `images.jsonl`
- `relations_chunk_asset.jsonl`
- `vector_store/doc_embeddings.npy`
- `vector_store/page_embeddings.npy`
- `vector_store/chunk_embeddings.npy`
- `vector_store/table_embeddings.npy`

## 评测数据模板

仓库提供了一个评测样例模板：

```text
soar/rag_eval_cases_template.jsonl
```

可以用它来组织检索问题、覆盖关键词、预期引用方式和答案完整性要求。

## 常见问题

### 启动 Web 应用时报数据库错误

优先检查：

- `soar/.env` 里的 `DB_MODE`
- `DATABASE_URL` 是否可用
- `SQLITE_PATH` 是否指向可写位置

### 检索脚本可以跑，但结果为空

优先检查：

- `--rag-dir` 是否指向正确目录
- 目标目录下是否存在 `docs.jsonl`、`pages.jsonl`、`chunks.jsonl`
- `vector_store/` 下的向量文件是否完整

### 重建流程提示缺少 API Key

这是正常保护行为。表格转换、关键词抽取和向量构建阶段都可能依赖 `SILICONFLOW_API_KEY`。

## 说明

这份 README 是根据仓库当前代码入口整理出来的使用文档。如果后续你调整了目录、模型提供方或启动方式，建议同步更新本文档。
