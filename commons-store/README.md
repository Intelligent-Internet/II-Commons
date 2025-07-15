# ii-commons-store: A Plug-and-Play Knowledge Hub for AI Applications

**An infrastructure project designed to enhance Large Language Models (LLMs) and Agents with real-time, reliable knowledge. We provide a growing collection of professional knowledge bases, starting with arXiv, to equip your AI applications with the latest scientific and technological insights.**

This project provides a high-performance semantic search and storage API. It allows you to store text data, generate vector embeddings for it, and perform semantic searches. It supports downloading [pre-computed vector data](DATASETS.md) directly from Hugging Face, eliminating the need for time-consuming local embedding calculations.

---

### The Challenge

*   **Factual Inaccuracy:** LLMs can "hallucinate" or generate incorrect information, a critical flaw for professional applications.
*   **Stale Knowledge:** An LLM's knowledge is frozen at its training date, leaving it unaware of recent discoveries and data.
*   **Missing Domain Expertise:** General models lack the specialized knowledge required for vertical domains like law, medicine, or finance.
*   **Inefficient Retrieval:** Live web searches by LLMs are often slow, costly, and produce unreliable results, failing to meet the demands of professional-grade applications.

### Our Solution

**ii-commons-store** offers a **Knowledge-as-a-Service** solution. We curate and pre-process high-quality information from reliable sources into standardized, ready-to-use **External Knowledge Bases**.

Our simple API empowers your LLM or Agent to query these knowledge bases, grounding its responses in verifiable facts and driving more reliable decision-making.

### Core Features

*   **Modular and Scalable Architecture**
    The architecture is designed to support multiple, independent knowledge bases. We start with **arXiv papers**, providing instant access to the latest scientific research, with plans to expand to other domains like PubMed.

*   **Plug-and-Play**
    Just run the service to create an instant knowledge hub. Agent queries are executed seamlessly, with no need for manual data or database management. The service can automatically handles downloading and loading in the background for a seamless experience.

*   **Open and Compatible**
    We support commercially-friendly, open-source embedding models, ensuring you're not locked into a specific vendor and can easily integrate with your existing tech stack.

### Key Advantages

*   **Improve Accuracy and Reliability**
    Reduce AI hallucinations by grounding responses in trusted sources like arXiv, ensuring answers are fact-based and verifiable.

*   **Empower Your Agents**
    Equip your Agents with powerful, out-of-the-box knowledge tools to significantly enhance their analytical and reasoning capabilities.

*   **Save Time and Reduce Costs**
    Eliminate the significant time and engineering costs of collecting and processing vast datasets like arXiv, allowing your team to focus on core application logic.

*   **Future-Proof**
    Our collection of knowledge bases is continually growing. Your Agent will automatically gain access to new domains as they are added.

## Quick Start

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

### Configuration

The project includes several sample configuration files. You need to create and modify them according to your needs.

Copy `api_server_config.yaml.sample` to `api_server_config.yaml` and edit it as needed. This file defines the server host, port, and dataset directory.

### Download Datasets

Please reference [DATASETS.md](DATASETS.md) for more information about datasets.

**Automatic Download:**

The API server is configured to automatically download the required dataset files (`.yaml` and `.duckdb`) upon startup if they are not found locally. This process uses the `datasets` list in your `api_server_config.yaml`. When you start the server, it will check for the existence of these files in the specified `search_config_directory` and download any that are missing.

**Manual Download (Optional):**

If you prefer to download the files manually before starting the server, or if you encounter issues with the automatic download, you can use the provided `download_hf_duckdb.py` script.

**Manual Download Steps:**

1.  Open your `api_server_config.yaml` file.
2.  Find the value of `search_config_directory`. This will be the `--output_dir` for your download script.
3.  For each item in the `datasets` list:
    -   `repo_id` corresponds to the `--repo_id` parameter.
    -   `name` is used to construct two filenames: `{name}.yaml` and `{name}.duckdb`. These two filenames will be passed to the `--filenames` parameter.

**Example:**

Assuming your `api_server_config.yaml` contains the following:

```yaml
search_config_directory: "data_dir"
datasets:
  - repo_id: "Intelligent-Internet/arxiv"
    name: "duckdb/arxiv_snowflake2m_128_int8"
```

You should run the following command to download the required files:

```bash
python download_hf_duckdb.py \
  --repo_id Intelligent-Internet/arxiv \
  --filenames duckdb/arxiv_snowflake2m_128_int8.yaml duckdb/arxiv_snowflake2m_128_int8.duckdb \
  --output_dir data_dir
```

*Please ensure that the `--output_dir` path exactly matches the `search_config_directory` you set in `api_server_config.yaml` so that the API server can find these files.*

### Run the API Server

Start the FastAPI application. Use the host and port configured in `api_server_config.yaml`:

```bash
python search_api_server.py --config api_server_config.yaml
```

Alternatively, you can use the JINA embedding API by passing the JINA API KEY as an environment variable:

```bash
JINA_KEY=jina_xxxx python search_api_server.py --config api_server_config.yaml
```

After the server starts, you can access the auto-generated API documentation at `http://127.0.0.1:5000/docs`.

## API Endpoints

The main API endpoints are as follows:

- `GET /configs`: List all loaded configurations
- `POST /search`: Perform a semantic search based on a query text.
- `POST /direct_search`: Perform a direct query on a database table.
- `POST /add`: Add a new text chunk to the database.
- `POST /process_embeddings`: Generate embeddings for pending text chunks in the database.
- `POST /manage_tags`: Add or remove tags for a specified record.
- `GET /health`: Check the health status of the API service.

### Usage Example

Call the search API using `curl`:

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

### Custom Search Configuration

If you are not using the downloaded datasets but want to build your own, or use this as a local RAG tool, model memory storage tool, etc., you can do so by defining your own search configuration. In the `data_dir/` directory, copy a sample configuration (e.g., `search_config.yaml.sample`) to your own configuration file (e.g., `my_search_config.yaml`). In this file, you can define the database path, embedding model, table structure, etc.
