# 溯源 · 商品资料智能问答系统（RAG 知识库）

基于 **FastAPI + LangGraph** 的商品资料 RAG 问答系统。把厂商的产品手册、操作说明、维修资料等非结构化文档转成可检索知识，支持自然语言提问，并让每个答案都能**回溯到手册的具体章节与原图**。

## 功能概览

| 模块 | 能力 |
| --- | --- |
| 文档导入 | PDF / Markdown 解析（MinerU）、图片语义补全、标题层级切分、商品名识别、稠密+稀疏双向量入库 |
| 智能问答 | 商品名确认与指代消解、三路并行召回、RRF 融合、Rerank 精排、断崖检测截断、SSE 流式输出 |
| 工程支撑 | 异步导入任务 + 进度轮询、节点级耗时日志、限流与多级降级 |

## 架构

拆分为**导入**与**查询**两个独立服务，各自用 LangGraph 编排：

```
导入服务 (:8000)
  POST /upload  -> 本地落盘 + MinIO -> BackgroundTasks
  GET  /status/{task_id} -> 前端轮询进度

  entry -> pdf_to_md -> md_img -> document_split
        -> item_name_rec -> bge_embedding -> import_milvus      (7 节点)

查询服务 (:8001)
  POST /query  |  GET /stream/{task_id} (SSE)  |  /history/{session_id}

  item_name_confirm
        +- multi_search -+-> search_embedding -+
                         +-> search_hyde       +-> join -> rrf -> rerank -> answer_output
                         +-> web_search_mcp    +
                       (9 节点，含 2 个编排用虚拟节点)
```

**存储分工**

- **Milvus**：`kb_chunks_v1`（切片 + 稠密/稀疏向量 + 溯源标量字段）、`kb_item_names_v1`（商品名向量，用于对齐）
- **MongoDB**：`chat_message`（多轮会话、改写后 query、item_names）
- **MinIO**：原始手册与手册内图片
- **本地磁盘**：解析中间产物（Markdown、`chunks.json`），便于排查

## 核心设计

1. **一个商品名多个向量入口** — 同一商品名按 `name / description / alias` 分别生成向量，payload 存完整信息，命中任一表述都能拿回全量数据。
2. **切分保语义** — 按 1–6 级标题切分并保留 `parent_title` 形成章节树；超长章节递归切、过短且同父标题的合并；代码围栏内的 `#` 不误判为标题。
3. **商品名确认前置** — 检索前先用 LLM 结合历史做指代消解与 query 改写，再与商品名向量库对齐，三态决策（直接确认 / 返回候选 / 引导补型号），确认结果作为下游 `item_name` 标量过滤条件，避免问 A 答 B。
4. **多路召回 + 动态截断** — 向量、HyDE、联网（MCP）三路并行；本地两路先按 `chunk_id` 做 RRF（k=60）去重融合，再与网页结果合并做 Rerank 精排；用断崖检测按分数落差动态决定保留条数（有上下界兜底），而不是固定 top_k。
5. **图片不丢失** — 图片取「章节标题 + 上下段」作为上下文调 VLM 生成摘要，原图转存 MinIO 后把 Markdown 里的本地路径回灌为对象地址，保证切分后图片仍能随答案展示。

## 技术栈

| 层 | 选型 |
| --- | --- |
| Web | FastAPI + Uvicorn |
| 编排 | LangGraph（`StateGraph` + 条件边） |
| 生成模型 | Qwen-Flash；图片理解 Qwen-VL-Flash |
| Embedding | BAAI/bge-m3（稠密 1024 维 + 稀疏） |
| Rerank | BAAI/bge-reranker-large |
| 向量库 | Milvus |
| 对象存储 | MinIO |
| 会话存储 | MongoDB |
| 文档解析 | MinerU |
| 联网兜底 | 阿里云百炼 WebSearch（MCP 协议） |
| 流式输出 | SSE |

## 目录结构

```
knowledge/
├── api/                      # 路由：import_router / query_router
├── processor/
│   ├── import_process/       # 导入图：main_graph + nodes/ + config + state
│   └── query_process/        # 查询图：main_graph + nodes/ + config + state
├── prompt/                   # import_prompt / query_prompt
├── schema/                   # 请求响应模型
├── service/                  # ImportFileService / QueryService
├── utils/                    # milvus / sse / mongo_history / task / embedding / markdown
├── front/                    # chat.html / import.html
├── core/                     # deps / paths
└── requirements.txt
```

## 配置

在项目根目录创建 `.env`（已被 gitignore，需自行准备），可先复制 `.env.example`：

```dotenv
# 模型
OPENAI_API_BASE=
OPENAI_API_KEY=
MODEL=qwen-flash
VL_MODEL=qwen3-vl-flash
ITEM_MODEL=qwen-flash

# Milvus
MILVUS_URL=http://localhost:19530
CHUNKS_COLLECTION=kb_chunks_v1
ITEM_NAME_COLLECTION=kb_item_names_v1
ENTITY_NAME_COLLECTION=kb_entity_names_v1
EMBEDDING_DIM=1024

# MinIO
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=
MINIO_SECRET_KEY=
MINIO_BUCKET_NAME=

# MongoDB
MONGO_URL=mongodb://localhost:27017
MONGO_DB_NAME=kb001

# MCP 联网检索
MCP_DASHSCOPE_BASE_URL=
```

切片长度、重叠句数、RRF 的 `k`、Rerank 的上下界与断崖阈值、商品名置信度等参数集中在
`knowledge/processor/import_process/config.py` 与 `knowledge/processor/query_process/config.py`。

## 启动

```bash
pip install -r requirements.txt

# 导入服务（:8000）
python -m knowledge.api.import_router

# 查询服务（:8001）
python -m knowledge.api.query_router
```

访问 `http://localhost:8001/front/chat.html` 进行问答，
`http://localhost:8000/front/import.html` 导入文档。

## 说明

- 个人项目，用于验证 RAG 检索链路的工程落地，非公司线上系统。
- **已知局限**
  - 知识库更新目前为全量重建（`_ensure_collection` 先删后建），后续可改为增量 upsert + 集合别名切换；
  - 评测依赖人工 badcase，尚未接入 RAGAS 离线回归；
  - 图谱抽取（实体名向量集合）为预留方向，当前未实现。