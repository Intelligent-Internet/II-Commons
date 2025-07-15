import duckdb
import hashlib
from typing import List, Optional, Dict, Any
from search_tool import get_db_connection

def _check_writable(config: Dict[str, Any]):
    if not config.get("database_writable"):
        raise ValueError("Database is not configured to be writable. Set 'database_writable: true' in the config.")

def initialize_database(config: Dict[str, Any]):
    _check_writable(config)
    db_file = config["db_file"]
    meta_table = config["meta_table_name"]
    emb_table = config["emb_table_name"]
    meta_id_col = config["meta_id_col"]
    emb_id_col = config["emb_id_col"]
    emb_vector_col = config["emb_vector_col"]

    print(f"Initializing database schema in: {db_file}")
    con = get_db_connection(db_file, read_only=False)
    con.execute(f"CREATE SEQUENCE IF NOT EXISTS seq_{meta_table} START 1;")
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {meta_table} (
            {meta_id_col} BIGINT PRIMARY KEY DEFAULT nextval('seq_{meta_table}'),
            doc_id VARCHAR UNIQUE,
            url VARCHAR UNIQUE,
            hash CHAR(64) UNIQUE,
            meta TEXT,
            tags VARCHAR[],
            chunk_text TEXT
        );
    """)
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {emb_table} (
            {emb_id_col} BIGINT PRIMARY KEY,
            {emb_vector_col} FLOAT[]
        );
    """)
    print(f"Database {db_file} initialized successfully.")

def add_text_chunk(config: Dict[str, Any], chunk_text: str, doc_id: str = None, url: str = None, meta: str = None, tags: list = None):
    _check_writable(config)
    if not chunk_text:
        raise ValueError("chunk_text cannot be empty")

    sha256_hash = hashlib.sha256(chunk_text.encode('utf-8')).hexdigest()
    db_file = config["db_file"]
    meta_table = config["meta_table_name"]
    con = get_db_connection(db_file, read_only=False)
    
    existing = con.execute(f"SELECT {config['meta_id_col']} FROM {meta_table} WHERE hash = ?", [sha256_hash]).fetchone()
    if existing:
        print(f"Text chunk with hash {sha256_hash} already exists (ID: {existing[0]}). Skipping.")
        return
        
    con.execute(f"""
        INSERT INTO {meta_table} (doc_id, url, hash, meta, tags, chunk_text)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (doc_id, url, sha256_hash, meta, tags, chunk_text))
    print(f"Text chunk with hash {sha256_hash} was successfully added to the database.")

def process_pending_embeddings(config: Dict[str, Any], model_instance: Any, task_type: str = ""):
    _check_writable(config)
    if model_instance is None:
        raise ValueError("A valid model instance must be provided.")

    db_file = config["db_file"]
    meta_table = config["meta_table_name"]
    emb_table = config["emb_table_name"]
    meta_id_col = config["meta_id_col"]
    emb_id_col = config["emb_id_col"]
    emb_vector_col = config["emb_vector_col"]

    con = get_db_connection(db_file, read_only=False)
    pending_rows = con.execute(f"""
        SELECT m.{meta_id_col}, m.chunk_text
        FROM {meta_table} m
        LEFT JOIN {emb_table} e ON m.{meta_id_col} = e.{emb_id_col}
        WHERE e.{emb_id_col} IS NULL
    """).fetchall()

    if not pending_rows:
        print("No text chunks found requiring embedding generation.")
        return

    print(f"Found {len(pending_rows)} text chunks to process for embeddings.")

    for row_id, chunk_text in pending_rows:
        try:
            embedding = model_instance.generate_embedding([chunk_text], task_type=task_type, mrl=None)
            if embedding is not None and len(embedding) > 0:
                con.execute(f"INSERT INTO {emb_table} ({emb_id_col}, {emb_vector_col}) VALUES (?, ?)", (row_id, embedding[0]))
                print(f"Successfully generated and stored embedding for ID {row_id}.")
            else:
                print(f"Warning: Embedding generation returned empty for ID {row_id}.")
        except Exception as e:
            print(f"Error generating embedding for ID {row_id}: {e}")
            continue

def manage_tags_for_record(config: Dict[str, Any], identifier: Dict[str, Any], tags_to_add: Optional[List[str]] = None, tags_to_remove: Optional[List[str]] = None):
    _check_writable(config)
    if not identifier or len(identifier) != 1:
        raise ValueError("Exactly one identifier (e.g., {'doc_id': 'some_id'}) must be provided.")
    
    identifier_key, identifier_value = list(identifier.items())[0]
    db_file = config["db_file"]
    meta_table = config["meta_table_name"]
    meta_id_col = config["meta_id_col"]
    con = get_db_connection(db_file, read_only=False)
    
    record = con.execute(f"SELECT {meta_id_col}, tags FROM {meta_table} WHERE {identifier_key} = ?", [identifier_value]).fetchone()

    if not record:
        print(f"No record found with {identifier_key} = '{identifier_value}'.")
        return False

    record_id, current_tags = record
    current_tags_set = set(current_tags if current_tags is not None else [])
    
    if tags_to_add:
        current_tags_set.update(tags_to_add)
    
    if tags_to_remove:
        current_tags_set.difference_update(tags_to_remove)
        
    new_tags_list = sorted(list(current_tags_set))

    if set(current_tags if current_tags else []) != set(new_tags_list):
        con.execute(f"UPDATE {meta_table} SET tags = ? WHERE {meta_id_col} = ?", (new_tags_list, record_id))
        print(f"Tags for record ID {record_id} updated to: {new_tags_list}")
    else:
        print(f"Tags for record ID {record_id} remain unchanged.")
    
    return True
