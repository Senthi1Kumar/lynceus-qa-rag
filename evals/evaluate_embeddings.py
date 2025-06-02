"""
This script evaluates the embedding and cross-encoder models' performance 
using the MLflow's evaluate method with various metrics like precision, reacall
and NDCG on synthetic queries and log results to MLflow.
"""

import json
import time
import logging
from typing import Dict, List, Any

import torch
import mlflow
import pandas as pd
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import CrossEncoder
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from hydra import initialize, compose
from omegaconf import DictConfig


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Hydra config
initialize(config_path='../conf/', job_name='app', version_base='1.3')
cfg: DictConfig = compose(config_name='config')

EVAL_DATA_PATH = cfg.eval.eval_qn_json
if not EVAL_DATA_PATH:
    logging.error("EVAL_DATA path not set (path to evaluation JSON file) in config.")
    raise ValueError("EVAL_DATA path not set.")

try:
    logging.info(f'Loading evaluation questions from: {EVAL_DATA_PATH}')
    eval_df = pd.read_json(EVAL_DATA_PATH)

    if eval_df.empty:
        logging.error(f"Evaluation data file '{EVAL_DATA_PATH}' is empty or not valid JSON.")
        raise ValueError("Evaluation data is empty.")
    
    # Validate required columns
    required_cols = ['generated_question', 'reference_node_ids']
    if not all(col in eval_df.columns for col in required_cols):
        logging.error(f"Evaluation data must contain columns: {required_cols}. Found: {eval_df.columns.tolist()}")
        raise ValueError("Missing required columns in evaluation data.")
    
    logging.info(f"Successfully loaded {len(eval_df)} evaluation entries.")
    
except Exception as e:
    logging.error(f'Failed to load or validate evaluation data from {EVAL_DATA_PATH}: {e}')
    raise

embedding_model_name = cfg.embedding_model.name
if not embedding_model_name:
    logging.error("EMBEDDING_MODEL not set in 'config.yaml'.")
    raise ValueError("EMBEDDING_MODEL not set in config.")

embed_dim = int(cfg.embedding_model.embed_dim)
if not embed_dim:
    logging.error("embed_dim not set in config.")
    raise ValueError("embed_dim not set in config.")

cross_encoder_model_name = cfg.cross_encoder.name

# Initialize models
try:
    embed_model = HuggingFaceEmbedding(
        model_name=embedding_model_name,
        max_length=embed_dim,
        cache_folder=cfg.hf_cache_dir,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    logging.info(f'Initialized embedding model: {embedding_model_name}')
except Exception as e:
    logging.error(f"Error initializing embedding model '{embedding_model_name}': {e}")
    raise

cross_encoder_model = None
if cross_encoder_model_name:
    try:
        cross_encoder_model = CrossEncoder(
            model_name=cross_encoder_model_name,
            # cache_dir=cfg.hf_cache_dir,
            # local_files_only=True,
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )
        logging.info(f"Initialized Re-ranker: Cross-Encoder model - {cross_encoder_model_name}")
    except Exception as e:
        logging.error(f"Error initializing Cross-Encoder model '{cross_encoder_model_name}': {e}")
else:
    logging.info("No CROSS_ENCODER_MODEL specified. Re-ranking will be skipped if attempted.")

pg = cfg.db
db_name = pg.db_name
db_host = pg.db_host
db_user = pg.db_user
db_pwd = pg.db_pwd
db_port = pg.db_port
table_name = cfg.table.name

if not all([db_name, db_host, db_user, db_pwd, table_name]):
    logging.error("Missing one or more PostgreSQL connection parameters in 'conf/db/postgres.yaml' file (db_name, db_host, db_user, db_pwd, table_name).")
    raise ValueError("Database connection parameters are not fully set.")

# HNSW Index Parameter
hnsw_ef_search = int(cfg.index.hnsw_ef_search)

# Re-ranking Parameter
rerank_multiplier = int(cfg.rerank.multiplier)

# MLflow Configuration
mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
logging.info(f"MLflow tracking URI set to: {cfg.mlflow.tracking_uri}")


def get_db_connection() -> psycopg2.extensions.connection:
    """Establishes and returns a new database connection."""
    try:
        conn = psycopg2.connect(
            dbname=db_name,
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_pwd
        )
        try:
            register_vector(conn)
            logging.debug("Registered vector typecaster for connection.")
        except ImportError:
            logging.warning("pgvector.psycopg2 not found. Ensure 'pgvector' is installed for this environment.")
        except Exception as e:
            logging.warning(f"Could not register vector type for connection: {e}")
        return conn
    except psycopg2.Error as e:
        logging.error(f"Error connecting to PostgreSQL database '{db_name}' on host '{db_host}': {e}")
        raise

def retrieve_doc_ids(query: str, top_k: int = 5, use_reranker: bool = False) -> List[str]:
    """
    Retrieves top_k document node_ids for a given query, optionally using a re-ranker.
    """
    if embed_model is None:
        logging.error("Global embedding model not initialized.")
        raise RuntimeError("Embedding model is not configured.")
    
    if use_reranker and cross_encoder_model is None:
        logging.warning("Re-ranking requested but cross-encoder model is not initialized. Proceeding without re-ranking.")
        use_reranker = False

    logging.debug(f"Generating embedding for query (first 50 chars): '{query[:50]}...'")
    query_embedding = embed_model.get_query_embedding(query)

    conn = None
    retrieved_node_ids: List[str] = []
    initial_candidates_for_reranking: List[Dict[str, Any]] = []

    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(f"SET LOCAL hnsw.ef_search = {hnsw_ef_search};")

            initial_limit = top_k * rerank_multiplier if use_reranker else top_k
            
            logging.debug(f"Executing initial vector search (limit={initial_limit}, ef_search={hnsw_ef_search})")

            search_sql = f"""
            SELECT node_id, text, metadata
            FROM {table_name}
            ORDER BY embedding <=> %s::vector({embed_dim})
            LIMIT %s;
            """
            cur.execute(search_sql, (query_embedding, initial_limit))
            initial_results = cur.fetchall()
            logging.debug(f"Retrieved {len(initial_results)} initial candidates from DB.")

            for row in initial_results:
                node_id, text_content, metadata_json = row
                try:
                    metadata_dict = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
                except json.JSONDecodeError:
                    logging.warning(f"Could not decode metadata for node {node_id}. Using raw string.")
                    metadata_dict = {"raw_metadata": metadata_json}

                initial_candidates_for_reranking.append({
                    "node_id": str(node_id),
                    "text": text_content,
                    "metadata": metadata_dict
                })
        
        if conn:
            conn.close()
            logging.debug("Database connection closed after initial retrieval.")

    except psycopg2.Error as db_err:
        logging.error(f"Database error during retrieval: {db_err}")
        if conn: 
            conn.close()
        return [] 
    except Exception as e:
        logging.error(f"An unexpected error occurred during retrieval: {e}")
        if conn: 
            conn.close()
        return []

    if not initial_candidates_for_reranking:
        logging.info('⚠️ No initial candidates retrieved.')
        return []

    # Re-ranking step
    if use_reranker:
        if not cross_encoder_model:
            logging.warning("Cross-encoder model not available for re-ranking. Returning top_k from initial retrieval.")
            retrieved_node_ids = [doc['node_id'] for doc in initial_candidates_for_reranking[:top_k]]
        else:
            logging.info(f"Starting re-ranking process on {len(initial_candidates_for_reranking)} candidates...")
            rerank_start_time = time.time()
            
            sentence_pairs = [(query, doc['text']) for doc in initial_candidates_for_reranking]
            
            try:
                rerank_scores = cross_encoder_model.predict(sentence_pairs, show_progress_bar=False, batch_size=1)
            except Exception as rerank_err:
                logging.error(f"Error during cross_encoder.predict: {rerank_err}")
                retrieved_node_ids = [doc['node_id'] for doc in initial_candidates_for_reranking[:top_k]]
                return retrieved_node_ids

            scored_docs = []
            for i, doc_data in enumerate(initial_candidates_for_reranking):
                doc_data["rerank_score"] = float(rerank_scores[i])
                scored_docs.append(doc_data)
            
            reranked_docs = sorted(scored_docs, key=lambda x: x['rerank_score'], reverse=True)
            
            rerank_end_time = time.time()
            logging.info(f"Re-ranking completed in {rerank_end_time - rerank_start_time:.2f} sec.")
            
            final_docs_after_reranking = reranked_docs[:top_k]
            retrieved_node_ids = [doc['node_id'] for doc in final_docs_after_reranking]
            logging.info(f"Returning {len(retrieved_node_ids)} docs after re-ranking.")
    else:
        retrieved_node_ids = [doc['node_id'] for doc in initial_candidates_for_reranking]
        logging.info(f"Returning {len(retrieved_node_ids)} docs without re-ranking.")
        
    return retrieved_node_ids


def evaluate_retrieval_pipeline(
        eval_dataframe: pd.DataFrame,
        use_reranker_flag: bool,
        top_k: int,
        experiment_name: str = "RAG_Retrieval_Evaluation"
    ):
    """
    Evaluates the retrieval pipeline (embedding model + optional re-ranker)
    using MLflow.
    """
    logging.info(f"Starting evaluation for embedding model: {embedding_model_name}, Re-ranking: {use_reranker_flag}")
    mlflow.set_experiment(experiment_name)

    def retriever_model_func(question_df: pd.DataFrame) -> pd.Series:
        retrieved_ids_series = question_df['generated_question'].apply(
            lambda q: retrieve_doc_ids(q, top_k=top_k, use_reranker=use_reranker_flag)
        )
        return retrieved_ids_series

    run_name_suffix = "with_reranker" if use_reranker_flag and cross_encoder_model else "without_reranker"
    with mlflow.start_run(run_name=f"Eval_{embedding_model_name}_{run_name_suffix}", log_system_metrics=True) as run:
        mlflow.log_param("embedding_model", embedding_model_name)
        mlflow.log_param('top_k', top_k)
        mlflow.log_param("re_ranking_enabled", use_reranker_flag)
        if use_reranker_flag and cross_encoder_model_name:
            mlflow.log_param("cross_encoder_model", cross_encoder_model_name)
        mlflow.log_param("hnsw_ef_search", hnsw_ef_search)
        if use_reranker_flag:
            mlflow.log_param("rerank_multiplier_for_initial_fetch", rerank_multiplier)
        
        logging.info(f"Starting mlflow.evaluate for run: {run.info.run_name}")
        
        # Ensure 'reference_node_ids' contains lists of strings
        eval_dataframe['reference_node_ids'] = eval_dataframe['reference_node_ids'].apply(
            lambda ids: [str(id_val) for id_val in ids] if isinstance(ids, list) else [str(ids)] if pd.notna(ids) else []
        )

        results = mlflow.evaluate(
            model=retriever_model_func,
            data=eval_dataframe,
            targets="reference_node_ids",
            model_type="retriever",
            evaluators="default"
        )
        logging.info("mlflow.evaluate completed.")
        logging.info(f"MLflow Evaluation Results:\n{results.metrics}")
        return results

if __name__ == "__main__":
    # Evaluate with re-ranking
    if cross_encoder_model:
        logging.info("\n--- Evaluating w/ Re-ranking ---")
        eval_results_with_reranker = evaluate_retrieval_pipeline(eval_df.copy(), use_reranker_flag=True, top_k=5)
    else:
        logging.warning("Cross-encoder model not initialized. Skipping evaluation with re-ranking.")

    # Evaluate without re-ranking
    logging.info("\n--- Evaluating w/o Re-ranking ---")
    eval_results_without_reranker = evaluate_retrieval_pipeline(eval_df.copy(), use_reranker_flag=False, top_k=5)

    logging.info("Evaluation process finished.")
