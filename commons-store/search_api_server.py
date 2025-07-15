import os
import argparse
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
from fastapi.responses import RedirectResponse
from download_hf_duckdb import download_file_from_hf

from search_tool import search_similar_documents, get_search_config_details, direct_search
from storage_tool import initialize_database, add_text_chunk, process_pending_embeddings, manage_tags_for_record

CONFIG_DIRECTORY_PATH = "data_dir/"
CACHED_SEARCH_CONFIGS: Dict[str, Optional[Dict[str, Any]]] = {}

class SearchRequest(BaseModel):
    config_name: Optional[str] = Field("duckdb_arxiv_snowflake2m_128_int8", description="The name of the search configuration to use.")
    query_text: str = Field("ai application in healthcare", description="The text to search for.")
    top_k: Optional[int] = Field(10, gt=0, le=100, description="Number of top results to return.")
    tags: Optional[str] = Field("", description="Optional comma-separated string of tags to filter results.")

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    config_name_used: str
    query_text: str
    top_k: int
    tags_applied: Optional[List[str]]

class AddTextRequest(BaseModel):
    config_name: str = Field(..., description="The name of the configuration whose database will be used.")
    chunk_text: str = Field(..., description="The text content to add.")
    doc_id: Optional[str] = None
    url: Optional[str] = None
    meta: Optional[str] = None
    tags: Optional[List[str]] = None

class ProcessEmbeddingsRequest(BaseModel):
    config_name: str = Field(..., description="The name of the configuration to process embeddings for.")

class ManageTagsRequest(BaseModel):
    config_name: str = Field(..., description="The name of the configuration whose database will be used.")
    identifier: Dict[str, Any] = Field(..., description="Identifier for the record (e.g., {'doc_id': '123'}).")
    tags_to_add: Optional[List[str]] = None
    tags_to_remove: Optional[List[str]] = None

class DirectSearchRequest(BaseModel):
    config_name: Optional[str] = Field("default", description="The name of the search configuration to use.")
    query_params: Dict[str, Any] = Field(..., description="A dictionary of column names and their desired values to search for.")
    limit: Optional[int] = Field(10, gt=0, le=100, description="Maximum number of results to return.")

class DirectSearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    config_name_used: str
    query_params: Dict[str, Any]

class ConfigInfo(BaseModel):
    config_name: str
    embedding_model: Optional[str] = Field(None, description="The embedding model ID used by the configuration.")

class ListConfigsResponse(BaseModel):
    configs: List[ConfigInfo] = Field(..., description="A list of available configurations.")

app = FastAPI(
    title="II-Commons-Store Semantic Search and Storage API",
    description="API for semantic search and data management in DuckDB.",
    version="0.2.0"
)

@app.get("/", include_in_schema=False)
async def root():
    """Redirects to the API documentation."""
    return RedirectResponse(url="/docs")


def load_and_prepare_configs():
    global CONFIG_DIRECTORY_PATH
    config_path = os.getenv("API_SERVER_CONFIG")
    if not config_path or not os.path.exists(config_path):
        print("Warning: API_SERVER_CONFIG not set. Using default config directory.")
        return
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            server_config = yaml.safe_load(f)

        config_parent_dir = os.path.dirname(os.path.abspath(config_path))

        # First, determine the search config directory
        search_config_dir = server_config.get("search_config_directory", "data_dir/")
        if not os.path.isabs(search_config_dir):
            search_config_dir = os.path.join(config_parent_dir, search_config_dir)
        CONFIG_DIRECTORY_PATH = search_config_dir

        # Now, check for datasets to download into that directory
        datasets_to_download = server_config.get("datasets")
        if datasets_to_download and isinstance(datasets_to_download, list):
            print("Checking for required datasets...")
            for dataset_info in datasets_to_download:
                repo_id = dataset_info.get("repo_id")
                name = dataset_info.get("name")

                if not all([repo_id, name]):
                    print(f"Warning: Skipping invalid dataset entry in server config: {dataset_info}")
                    continue

                files_to_check = [f"{name}.yaml", f"{name}.duckdb"]
                for filename in files_to_check:
                    # Download directory is the search_config_directory
                    output_dir = CONFIG_DIRECTORY_PATH
                    target_file_path = os.path.join(output_dir, filename)

                    if os.path.exists(target_file_path):
                        print(f"Dataset file '{os.path.basename(filename)}' already exists in '{output_dir}'. Skipping.")
                    else:
                        print(f"Dataset '{filename}' from repo '{repo_id}' not found locally. Downloading...")
                        download_file_from_hf(
                            repo_id=repo_id,
                            filename=filename,
                            output_dir=output_dir,
                            repo_type=dataset_info.get("repo_type", "dataset"),
                            token=os.environ.get("HUGGINGFACE_HUB_TOKEN"),
                            revision=dataset_info.get("revision", "main")
                        )
    except Exception as e:
        print(f"ERROR: Could not load server config '{config_path}': {e}")

@app.on_event("startup")
async def startup_event():
    global CACHED_SEARCH_CONFIGS, CONFIG_DIRECTORY_PATH
    load_and_prepare_configs()
    if not os.path.isdir(CONFIG_DIRECTORY_PATH):
        print(f"FATAL ERROR: Config directory '{CONFIG_DIRECTORY_PATH}' not found.")
        return
    print(f"Scanning for configs in: '{CONFIG_DIRECTORY_PATH}'")
    for dirpath, _, filenames in os.walk(CONFIG_DIRECTORY_PATH):
        for filename in filenames:
            if filename.endswith((".yaml", ".yml")):
                full_config_path = os.path.join(dirpath, filename)
                relative_path = os.path.relpath(full_config_path, CONFIG_DIRECTORY_PATH)
                config_name = os.path.splitext(relative_path)[0].replace(os.sep, '_')
                try:
                    search_config = get_search_config_details(full_config_path)
                    CACHED_SEARCH_CONFIGS[config_name] = search_config
                    if search_config:
                        print(f"Loaded config '{config_name}'.")
                        if search_config.get('database_writable'):
                            initialize_database(config=search_config)
                        model_id = search_config.get("emb_model_id")
                        if model_id:
                            from embedding_utils import get_embedding_provider
                            provider = get_embedding_provider(model_id=model_id, model_config=search_config.get("model_config_params"))
                            search_config["_cached_model_object"] = provider
                            print(f"Model provider for '{model_id}' loaded for '{config_name}'.")
                except Exception as e:
                    print(f"Error during startup for config '{config_name}': {e}")
                    CACHED_SEARCH_CONFIGS[config_name] = None

def get_config_and_model(config_name: str):
    if config_name == "default":
        name, cfg = next(((n, c) for n, c in CACHED_SEARCH_CONFIGS.items() if c), (None, None))
        if not cfg:
            raise HTTPException(status_code=404, detail="No default configuration available.")
        config_name, search_config = name, cfg
    else:
        search_config = CACHED_SEARCH_CONFIGS.get(config_name)
        if not search_config:
            raise HTTPException(status_code=404, detail=f"Configuration '{config_name}' not found.")
    return config_name, search_config, search_config.get("_cached_model_object")

@app.post("/search", response_model=SearchResponse, summary="Perform Semantic Search")
async def perform_search(request: SearchRequest):
    config_name, search_config, model_instance = get_config_and_model(request.config_name)
    
    tags_list = [tag.strip() for tag in request.tags.split(',')] if request.tags else None

    try:
        results = search_similar_documents(
            query_text=request.query_text, 
            search_config_details=search_config, 
            top_k=request.top_k, 
            tags=tags_list, 
            model_instance=model_instance
        )
        return SearchResponse(
            results=results, 
            config_name_used=config_name, 
            query_text=request.query_text, 
            top_k=request.top_k, 
            tags_applied=tags_list
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/direct_search", response_model=DirectSearchResponse, summary="Perform a Direct Search on Metadata")
async def perform_direct_search(request: DirectSearchRequest):
    config_name, search_config, _ = get_config_and_model(request.config_name)
    try:
        results = direct_search(search_config_details=search_config, query_params=request.query_params, limit=request.limit)
        return DirectSearchResponse(results=results, config_name_used=config_name, query_params=request.query_params)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add", summary="Add a Text Chunk")
async def add_text(request: AddTextRequest):
    _, search_config, _ = get_config_and_model(request.config_name)
    try:
        add_text_chunk(config=search_config, chunk_text=request.chunk_text, doc_id=request.doc_id, url=request.url, meta=request.meta, tags=request.tags)
        return {"status": "success", "message": "Text chunk added or already exists."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_embeddings", summary="Process Pending Embeddings")
async def process_embeddings_endpoint(request: ProcessEmbeddingsRequest):
    _, search_config, model_instance = get_config_and_model(request.config_name)
    if not model_instance:
        raise HTTPException(status_code=400, detail=f"No model configured for '{request.config_name}'.")
    try:
        process_pending_embeddings(config=search_config, model_instance=model_instance, task_type=search_config.get('query_task_type', ''))
        return {"status": "success", "message": "Embedding processing initiated."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/manage_tags", summary="Manage Tags for a Record")
async def manage_tags(request: ManageTagsRequest):
    _, search_config, _ = get_config_and_model(request.config_name)
    try:
        success = manage_tags_for_record(config=search_config, identifier=request.identifier, tags_to_add=request.tags_to_add, tags_to_remove=request.tags_to_remove)
        return {"status": "success" if success else "not_found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/configs", response_model=ListConfigsResponse, summary="List All Loaded Configurations")
async def list_configs():
    """
    Lists all successfully loaded search configurations and their associated embedding models.
    """
    configs_info = [
        ConfigInfo(config_name=name, embedding_model=config.get("emb_model_id"))
        for name, config in CACHED_SEARCH_CONFIGS.items() if config
    ]
    return ListConfigsResponse(configs=configs_info)

@app.get("/health", summary="Health Check")
async def health_check():
    parsed = [name for name, cfg in CACHED_SEARCH_CONFIGS.items() if cfg]
    failed = [name for name, cfg in CACHED_SEARCH_CONFIGS.items() if not cfg]
    status = "unhealthy" if not parsed else ("degraded" if failed else "healthy")
    return {"status": status, "parsed_configs": parsed, "failed_configs": failed}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DuckDB API Server")
    parser.add_argument("--config", type=str, default="api_server_config.yaml", help="Path to the server config file.")
    args = parser.parse_args()
    os.environ["API_SERVER_CONFIG"] = os.path.abspath(args.config)
    host, port = "127.0.0.1", 5000
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            server_cfg = yaml.safe_load(f).get("server", {})
            host = server_cfg.get("host", "127.0.0.1")
            port = server_cfg.get("port", 5000)
    except Exception as e:
        print(f"Warning: Could not read server host/port. Using defaults. Error: {e}")
    uvicorn.run("search_api_server:app", host=host, port=port, reload=True)
