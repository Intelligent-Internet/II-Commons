# ii-commons-store - 为你的 AI 应用提供即插即用的专业知识大脑

**一个为大语言模型（LLM）和智能体（Agent）提供实时、可信知识增强能力的基础设施项目。我们提供一个不断扩充的专业知识库系列，首个版本聚焦于 arXiv 数据库，为你的AI应用注入前沿科技知识。**


本项目提供高性能语义搜索和存储 API。它允许您存储文本数据，为其生成向量嵌入，并执行高效的语义搜索。支持直接从 huggingface 下载[预计算好的向量数据](DATASETS_cn.md)，无需本地进行耗时的嵌入计算。

---

###  问题

*   **事实性不足：** LLM 容易产生“幻觉”，在关键问题上编造事实，这在专业领域是不可接受的。
*   **知识陈旧：** 模型的内部知识有截止日期，无法获取特定领域的最新进展（如最新的科学论文、财经报告）。
*   **领域知识匮乏：** 通用模型在法律、医疗、金融等垂直领域的知识深度和广度均不足。
*   **信息获取低效：** 让LLM直接联网搜索，结果质量参差不齐、速度慢且成本不可控，难以满足专业应用对稳定性的要求。

###  我们的解决方案

**ii-commons-store** 提供了一种全新的**知识即服务（Knowledge-as-a-Service）**模式。我们预先将来自权威来源的高质量知识制作成标准化的**外置知识库（External Knowledge Bases）**。

我们赋予你的LLM或Agent**查询这些权威知识库**的能力。通过一个简单的API，Agent可以利用我们提供的知识工具，从而用可信的事实来驱动它的决策和回答。

###  核心特点

*    **模块化与可扩展的知识库架构**
    我们的核心架构是为支持多个独立的知识库而设计的。当前版本提供了 **arXiv 知识库**，让你的AI能即时访问最新的科研论文。未来，我们会基于此架构轻松地扩展到更多领域（如PubMed等）。


*    **即插即用**
    你只需运行我们的服务，它就会成为一个随时待命的知识中心。Agent发出的知识查询指令会被无缝执行，无需你手动管理任何数据或数据库。我们的服务会**按需、自动地**处理该知识库和embedding模型的下载与加载。

*    **开放与兼容 (Open & Compatible)**
    我们支持商业友好的开源Embedding模型。你不会被任何单一的技术或供应商锁定，可以灵活地集成到你现有的技术栈中。

###  主要优势 (Key Advantages)

*   **精准可靠**
    通过从可信数据源（arXiv）中检索信息，减少AI幻觉，让回答有据可查。

*   **赋能智能体**
    为Agent提供了强大的、开箱即用的知识工具，使其在科学技术领域的分析和回答能力得到质的飞跃。

*   **节约成本**
    为你节省了自行收集、处理海量arXiv论文所需的大量时间和工程成本。让你的团队可以专注于Agent的核心逻辑。

*   **面向未来创建**
    我们会逐步提供更多知识库。你的Agent将自动获得访问更多知识领域的能力。


## 快速开始

### Linux / Mac

```bash
git clone https://github.com/Intelligent-Internet/ii-commons-store-dev.git
cd ii-commons-store-dev
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp api_server_config.yaml.sample api_server_config.yaml
python search_api_server.py --config api_server_config.yaml
```

### Windows

```bash
git clone https://github.com/Intelligent-Internet/ii-commons-store-dev.git
cd ii-commons-store-dev
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
copy api_server_config.yaml.sample api_server_config.yaml
python search_api_server.py --config api_server_config.yaml
```

### 配置


**API 服务器配置**: 复制 `api_server_config.yaml.sample` 到 `api_server_config.yaml` 并根据需要进行编辑。此文件定义了服务器主机、端口和数据集目录。

### 下载数据集

关于数据集的说明，请参考： [DATASETS_cn.md](DATASETS_cn.md)

**自动下载:**

API 服务器被配置为在启动时自动下载所需的数据集文件（`.yaml` 和 `.duckdb`），如果这些文件在本地不存在的话。这个过程会使用您在 `api_server_config.yaml` 中定义的 `datasets` 列表。当您启动服务器时，它会检查在指定的 `search_config_directory` 目录中是否存在这些文件，并下载任何缺失的文件。

**手动下载（可选）:**

如果您希望在启动服务器前手动下载文件，或者在自动下载过程中遇到问题，您可以使用提供的 `download_hf_duckdb.py` 脚本。

**下载步骤:**

1.  打开您的 `api_server_config.yaml` 文件。
2.  找到 `search_config_directory` 的值。这将是您下载脚本中的 `--output_dir`。
3.  对于 `datasets` 列表中的每一个项目：
    -   `repo_id` 对应 `--repo_id` 参数。
    -   `name` 用于构建两个文件名：`{name}.yaml` 和 `{name}.duckdb`。这两个文件名将传递给 `--filenames` 参数。

**示例:**

假设您的 `api_server_config.yaml` 内容如下:

```yaml
search_config_directory: "data_dir"
datasets:
  - repo_id: "Intelligent-Internet/arxiv"
    name: "duckdb/arxiv_snowflake2m_128_int8"
```

您应该运行以下命令来下载所需文件：

```bash
python download_hf_duckdb.py \
  --repo_id Intelligent-Internet/arxiv \
  --filenames duckdb/arxiv_snowflake2m_128_int8.yaml duckdb/arxiv_snowflake2m_128_int8.duckdb \
  --output_dir data_dir
```

*请确保 `--output_dir` 的路径与您在 `api_server_config.yaml` 中设置的 `search_config_directory` 完全匹配，这样 API 服务器才能找到这些文件。*

### 运行 API 服务器

启动 FastAPI 应用。请使用在 `api_server_config.yaml` 中配置的主机和端口：

```bash
python search_api_server.py --config api_server_config.yaml
```

或者通过环境变量传入JINA API KEY来使用JINA embedding API

```bash
JINA_KEY=jina_xxxx python search_api_server.py --config api_server_config.yaml
```

服务器启动后，您可以在 `http://127.0.0.1:5000/docs` 访问自动生成的 API 文档。

## API 端点

以下是主要的 API 端点：

- `GET /configs`: 列出所有已加载的配置。
- `POST /search`: 根据查询文本执行语义搜索。
- `POST /direct_search`: 对数据库表执行直接查询。
- `POST /add`: 添加一个新的文本块到数据库。
- `POST /process_embeddings`: 为数据库中待处理的文本块生成嵌入。
- `POST /manage_tags`: 为指定的记录添加或移除标签。
- `GET /health`: 检查 API 服务的健康状态。

### 使用示例

使用 `curl` 调用搜索 API：

```bash
curl -X 'POST' \
  'http://127.0.0.1:5000/search' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "config_name": "arxiv_abstract_snowflake2_128_int8",
  "query_text": "healthcare AI applications",
  "top_k": 5
}'
```

6.  **自定义搜索配置**:
    如果不使用下载数据集，而是构建自己的数据集，或者作为本地RAG工具，模型记忆存储工具等用途。可以通过定义自己的搜索配置实现。在 `data_dir/` 目录下，复制一个示例配置（如 `search_config.yaml.sample`）为您自己的配置文件（例如 `my_search_config.yaml`）。您可以在此文件中定义数据库路径、嵌入模型、表结构等。

