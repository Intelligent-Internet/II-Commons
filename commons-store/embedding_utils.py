import numpy as np
import numba
from light_embed import TextEmbedding
from typing import Optional, Dict, Any, Union
from numpy.typing import NDArray
import requests
import os
from abc import ABC, abstractmethod


# 8bit/4bit scalar_quantize function from https://github.com/Snowflake-Labs/arctic-embed/blob/main/compressed_embeddings_examples/score_arctic_embed_m_v1dot5_with_quantization.ipynb with apache 2.0 license
@numba.njit(error_model="numpy", parallel=True)
def fast_8bit_uniform_scalar_quantize(
    emb_matrix: NDArray[np.float32], limit: float
) -> NDArray[np.uint8]:
    num_row, num_col = emb_matrix.shape
    assert limit > 0
    out = np.empty((num_row, num_col), dtype=np.uint8)
    bin_width = 2 * limit / 255
    for i in numba.prange(num_row):
        for j in range(num_col):
            out[i, j] = round(max(0, min(2 * limit, limit + emb_matrix[i, j])) / bin_width)
    return out

@numba.njit(error_model="numpy", parallel=True)
def fast_4bit_uniform_scalar_quantize(
    emb_matrix: NDArray[np.float32], limit: float
) -> NDArray[np.uint8]:
    num_row, num_col = emb_matrix.shape
    assert num_col % 2 == 0
    assert limit > 0
    out = np.empty((num_row, num_col // 2), dtype=np.uint8)
    bin_width = 2 * limit / 15
    for i in numba.prange(num_row):
        row = emb_matrix[i, :]
        for out_j in range(num_col // 2):
            # Pull out two values at a time.
            in_j = out_j * 2
            value1 = row[in_j]
            value2 = row[in_j + 1]

            # 4-bit quantize the values.
            value1 = round(max(0, min(2 * limit, limit + value1)) / bin_width)
            value2 = round(max(0, min(2 * limit, limit + value2)) / bin_width)

            # Pack the values into a single uint8.
            value_packed = (value1 << 4) | value2
            out[i, out_j] = value_packed
    return out

def l2_normalize_numpy_pytorch_like(arr, axis, epsilon=1e-12):
    if not isinstance(arr, np.ndarray):
        raise TypeError("Input must be a NumPy ndarray.")
    if axis is None:
        raise ValueError("Parameter 'axis' must be specified for PyTorch-like normalization.")

    if np.issubdtype(arr.dtype, np.integer):
        arr_float = arr.astype(np.float32)
    elif np.issubdtype(arr.dtype, np.floating):
        arr_float = arr.astype(np.promote_types(arr.dtype, np.float32))
    else: 
        arr_float = arr.astype(np.float32)

    norm = np.linalg.norm(arr_float, ord=2, axis=axis, keepdims=True)
    normalized_arr = arr_float / np.maximum(norm, epsilon)
    return normalized_arr

class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    @abstractmethod
    def generate_embedding(self, texts: list[str], task_type: str = "", mrl: Optional[int] = 128) -> np.ndarray:
        """Generates embeddings for a list of texts."""
        pass

class LocalModelProvider(EmbeddingProvider):
    """Embedding provider for local TextEmbedding models."""
    def __init__(self, model_id: str, model_config: Optional[Dict[str, Any]] = None):
        print(f"Loading new local model: {model_id} with config: {model_config is not None}")
        text_embedding_config = model_config.copy() if model_config else {}
        # Filter out API-specific keys that might be in the config
        text_embedding_config.pop("api_model_name", None)
        text_embedding_config.pop("api_key_env_var", None)

        if text_embedding_config:
            self.model = TextEmbedding(model_name_or_path=model_id, model_config=text_embedding_config)
        else:
            self.model = TextEmbedding(model_name_or_path=model_id)
        print(f"Model loaded: {model_id}")

    def generate_embedding(self, texts: list[str], task_type: str = "", mrl: Optional[int] = 128) -> np.ndarray:
        if not texts:
            return np.array([])

        # Logic from the old generate_embedding for local models
        processed_texts = []
        is_snowflake_model = ("snowflake-arctic-embed-m-v2.0" in self.model.model_name_or_path.lower() or
                               "snowflake-arctic-embed-l-v2.0" in self.model.model_name_or_path.lower())
        
        for text_content in texts:
            current_text = str(text_content) if not isinstance(text_content, str) else text_content
            if is_snowflake_model and task_type == "query":
                processed_texts.append(f"query: {current_text}")
            else:
                processed_texts.append(current_text)

        extra_kwargs = {}
        is_jina_model_local = "jina-embeddings-v3" in self.model.model_name_or_path.lower()
        if is_jina_model_local:
            config_lora_adaptations = self.model._transformer_config.get("lora_adaptations")
            effective_task_type = task_type if task_type else "retrieval.passage"
            if config_lora_adaptations and effective_task_type in config_lora_adaptations:
                task_id = np.array(config_lora_adaptations.index(effective_task_type), dtype=np.int64)
                extra_kwargs["task_id"] = task_id
            else:
                print(f"warning: Local Jina task '{effective_task_type}' invalid, using default task retrieval.passage")
                if config_lora_adaptations and "retrieval.passage" in config_lora_adaptations:
                    task_id = np.array(config_lora_adaptations.index("retrieval.passage"), dtype=np.int64)
                    extra_kwargs["task_id"] = task_id
                else:
                    print(f"warning: Default task 'retrieval.passage' not found in lora_adaptations for local Jina model.")

        embedding_array = self.model.encode(processed_texts, extra_kwargs=extra_kwargs)

        if mrl is None:
            return embedding_array

        target_dim = mrl
        if not isinstance(target_dim, int) or target_dim not in [128, 256]:
            print(f"warning: invalid mrl '{target_dim}'. using default 128")
            target_dim = 128

        if embedding_array.shape[1] > target_dim:
            embedding_array_processed = embedding_array[:, :target_dim]
        else:
            if embedding_array.shape[1] < target_dim:
                print(f"warning: model output dims {embedding_array.shape[1]} < mrl {target_dim}")
            embedding_array_processed = embedding_array
                
        normalized_embedding_array = l2_normalize_numpy_pytorch_like(embedding_array_processed, axis=1, epsilon=1e-12)
        return normalized_embedding_array

class JinaAPIProvider(EmbeddingProvider):
    """Embedding provider for the Jina AI API."""
    JINA_API_URL = 'https://api.jina.ai/v1/embeddings'
    JINA_API_DEFAULT_MODEL = "jina-embeddings-v3"
    JINA_API_DEFAULT_KEY_ENV_VAR = "JINA_KEY"

    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        config = model_config or {}
        self.api_model_name = config.get("api_model_name", self.JINA_API_DEFAULT_MODEL)
        self.api_key_env_var = config.get("api_key_env_var", self.JINA_API_DEFAULT_KEY_ENV_VAR)
        self.api_url = self.JINA_API_URL
        print(f"Jina API provider configured for model: {self.api_model_name}")

    def generate_embedding(self, texts: list[str], task_type: str = "", mrl: Optional[int] = 128) -> np.ndarray:
        if not texts:
            return np.array([])

        # Logic from the old generate_embedding for Jina API
        print(f"Generating embeddings via Jina API: {self.api_model_name}")
        api_key = os.environ.get(self.api_key_env_var)
        if not api_key:
            raise ValueError(f"Jina API key not found in environment variable '{self.api_key_env_var}'.")

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        
        jina_task = task_type if task_type else "retrieval.passage"
        data = {"model": self.api_model_name, "input": texts}
        if jina_task:
            data["task"] = jina_task
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            
            response_data = response.json()
            if "data" not in response_data or not isinstance(response_data["data"], list):
                raise ValueError("Jina API response format error: 'data' field missing or not a list.")

            embeddings_list = [item["embedding"] for item in response_data["data"]]
            if not embeddings_list:
                return np.array([])

            embedding_array = np.array(embeddings_list, dtype=np.float32)
            
            if mrl:
                embedding_array_processed = embedding_array[:, :mrl]
                normalized_embedding_array = l2_normalize_numpy_pytorch_like(embedding_array_processed, axis=1, epsilon=1e-12)
                return normalized_embedding_array
            else:
                return embedding_array

        except requests.exceptions.RequestException as e:
            print(f"Jina API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Jina API response content: {e.response.text}")
            return np.array([])
        except Exception as e:
            print(f"Error processing Jina API response: {e}")
            return np.array([])

_PROVIDER_CACHE = {}
_DEFAULT_MODEL_PATH = 'Snowflake/snowflake-arctic-embed-m-v2.0'

def get_embedding_provider(model_id: str = None, model_config: Optional[Dict[str, Any]] = None) -> EmbeddingProvider:
    """
    Gets a cached or new embedding provider instance.

    Args:
        model_id (str, optional): Model name, path, or "jina-api". Defaults to _DEFAULT_MODEL_PATH.
        model_config (dict, optional): Configuration for the model or API.

    Returns:
        EmbeddingProvider: An instance of a provider.
    """
    global _PROVIDER_CACHE
    actual_model_id = model_id if model_id is not None else _DEFAULT_MODEL_PATH
    
    # Create a cache key from model_id and sorted model_config items
    config_tuple = tuple(sorted(model_config.items())) if model_config else None
    cache_key = (actual_model_id, config_tuple)

    if cache_key in _PROVIDER_CACHE:
        print(f"Using cached provider for: {actual_model_id} with config: {model_config is not None}")
        return _PROVIDER_CACHE[cache_key]

    if actual_model_id == "jina-api":
        provider = JinaAPIProvider(model_config)
    else:
        provider = LocalModelProvider(actual_model_id, model_config)
    
    _PROVIDER_CACHE[cache_key] = provider
    return provider




if __name__ == '__main__':
    import torch
    
    # Get providers using the new factory function
    jina_provider = get_embedding_provider(model_id="jinaai/jina-embeddings-v3")
    snowflake_provider = get_embedding_provider(model_id='Snowflake/snowflake-arctic-embed-m-v2.0')
    
    #再次加载相同的模型，应该会使用缓存
    jina_provider_cached = get_embedding_provider(model_id="jinaai/jina-embeddings-v3")
    assert jina_provider is jina_provider_cached, "Jina provider should be cached"

    # 测试 Jina API (确保设置了 JINA_API_KEY 环境变量)
    # export JINA_API_KEY="your_jina_api_key_here"
    print("\n--- Testing Jina API ---")
    JINA_API_DEFAULT_KEY_ENV_VAR = "JINA_KEY" # Re-define for main block scope
    jina_api_key_present = os.environ.get(JINA_API_DEFAULT_KEY_ENV_VAR)
    if jina_api_key_present:
        jina_api_provider = get_embedding_provider(model_id="jina-api")
        
        api_test_texts = ["Hello from Jina API", "Another text for Jina API"]
        print(f"Test batch text for Jina API (task_type='retrieval.query'): {api_test_texts}")
        
        api_embeddings = jina_api_provider.generate_embedding(api_test_texts, task_type="retrieval.query")
        
        if api_embeddings.size > 0:
            print(f"Jina API batch embedding shape: {api_embeddings.shape}")
            for i, text_input in enumerate(api_test_texts):
                embedding_vector = api_embeddings[i]
                print(f"\nText (Jina API): \"{text_input}\"")
                print(f"First 5 vector: {embedding_vector[:5]}")
                print(f"Embedding dims: {len(embedding_vector)}")
        else:
            print("Failed to get embeddings from Jina API. Check API key and network.")
    else:
        print(f"Skipping Jina API test: Environment variable {JINA_API_DEFAULT_KEY_ENV_VAR} not set.")
    print("--- End of Jina API Test ---")


    test_texts_batch = [
        "这是一个示例文本，用于测试嵌入生成。",
        "另一个测试句子。",
        "",  # 空字符串
        "   ", # 仅空白字符的字符串
        "第三个非空句子用于测试。"
    ]
    
    # 使用 Jina provider 测试
    print(f"\nTesting with Jina provider")
    print(f"Test batch text (task_type='retrieval.query', mrl=128 - 默认): {test_texts_batch}")
    batch_embeddings_128_jina = jina_provider.generate_embedding(test_texts_batch, task_type="retrieval.query")
    
    print(f"Batch embedding shape (Jina, mrl=128): {batch_embeddings_128_jina.shape}")
    for i, text_input in enumerate(test_texts_batch):
        embedding_vector = batch_embeddings_128_jina[i]
        print(f"\nText (Jina, mrl=128): \"{text_input}\"")
        print(f"First 5 vector: {embedding_vector[:5]}")
        print(f"Embedding dims: {len(embedding_vector)}")

    # 使用 Snowflake provider 测试
    print(f"\nTesting with Snowflake provider")
    print(f"Test batch text (task_type='query', mrl=256): {test_texts_batch}")
    batch_embeddings_256_sf = snowflake_provider.generate_embedding(test_texts_batch, task_type="query", mrl=256)
    print(f"Batch embedding shape (Snowflake, mrl=256): {batch_embeddings_256_sf.shape}")
    for i, text_input in enumerate(test_texts_batch):
        embedding_vector = batch_embeddings_256_sf[i]
        print(f"\nText (Snowflake, mrl=256): \"{text_input}\"")
        print(f"First 5 vector: {embedding_vector[:5]}")
        print(f"Embedding dims: {len(embedding_vector)}")

    print(f"\nTest batch text (Snowflake, task_type='', mrl=None - 原始输出): {test_texts_batch}")
    batch_embeddings_none_sf = snowflake_provider.generate_embedding(snowflake_provider, test_texts_batch, task_type="", mrl=None)
    print(f"Batch embedding shape (Snowflake, mrl=None): {batch_embeddings_none_sf.shape}")
    for i, text_input in enumerate(test_texts_batch):
        embedding_vector = batch_embeddings_none_sf[i]
        print(f"\nText (Snowflake, mrl=None): \"{text_input}\"")
        print(f"First 5 vector: {embedding_vector[:5]}")
        print(f"Embedding dims: {len(embedding_vector)}")


    # 现有Jina模型测试部分 - 使用已加载的 jina_provider
    queries = ['query: what is snowflake?', 'query: Where can I get the best tacos?']
    documents = ['The Data Cloud!', 'Mexico City of Course!']
    
    print("\nTest document embedding (Jina, task_type='retrieval.passage', mrl=256):")
    document_embeddings_np = jina_provider.generate_embedding(documents, task_type='retrieval.passage', mrl=256)
    print(f"Document embedding shape (Jina, mrl=256): {document_embeddings_np.shape}")

    print("\nTest query embedding (Jina, task_type='retrieval.query', mrl=256):")
    query_embeddings_np = jina_provider.generate_embedding(queries, task_type='retrieval.query', mrl=256)
    print(f"Query embedding shape (Jina, mrl=256): {query_embeddings_np.shape}")
    
    if isinstance(query_embeddings_np, np.ndarray) and isinstance(document_embeddings_np, np.ndarray) \
       and query_embeddings_np.size > 0 and document_embeddings_np.size > 0:
        scores = torch.mm(torch.from_numpy(query_embeddings_np), torch.from_numpy(document_embeddings_np).transpose(0, 1))
        for query_idx, query_text in enumerate(queries):
            query_scores = scores[query_idx]
            doc_score_pairs = list(zip(documents, query_scores.tolist()))
            doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
            print("Query:", query_text)
            for document, score in doc_score_pairs:
                print(score, document)
