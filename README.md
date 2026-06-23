# new-rag-project
### 现有系统现状

| 组件 | 当前技术栈 |
|------|------------|
| 向量数据库 | SpaceRAG_v2 |
| Embedding | qwen3-vl(4096维) |
| 检索方式 | 向量检索 + BM25 + Alpha加权融合 |
| 重排序 | SiliconFlow Qwen3-Reranker-8B |
| 生成模型 | DashScope qwen-plus |
| 知识库 | 多行业分类，支持文档/图片/表格 |
