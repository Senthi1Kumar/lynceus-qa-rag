"""
This script connects with PostgreSQL DB and retrieves relevant documents
for a given query from the stored table using similarity search with 
`ef_search` parameter, then it reranks the relevant candidate documents 
with Cross-Encoder model and returns the top-k chunks.
"""

import gc
import json
import time
import logging
import hydra
from omegaconf import DictConfig
from typing import List, Dict, Any

import torch
import psycopg2
from pgvector.psycopg2 import register_vector
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers import CrossEncoder

logging.basicConfig(level=logging.INFO)

def get_db_connection(cfg: DictConfig):
    """Establishes and returns a new database connection."""
    try:
        pg = cfg.db
        conn = psycopg2.connect(
            dbname=pg.db_name,
            host=pg.db_host,
            port=pg.db_port,
            user=pg.db_user,
            password=pg.db_pwd
        )

        try:
            register_vector(conn)
            logging.info("Registered vector typecaster for connection.")
        except ImportError:
            logging.warning("pgvector.psycopg2 not found. Ensure 'pgvector' is installed.")
        except Exception as e:
            logging.warning(f"Could not register vector type for connection: {e}")

        return conn
    except psycopg2.Error as e:
        logging.error(f"Error connecting to PostgreSQL: {e}")
        raise


def retrieve_relevant_docs(
        cfg: DictConfig,
        query_text: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
    """
    Embeds the query and retrieves the top_k most relevant document chunks
    from the PostgreSQL using vector similarity search.
    """
    table_name = cfg.table.name

    # HNSW Query-Time Parameter
    # 'ef_search' controls the search time vs. accuracy trade-off. Higher = better accuracy, slower.
    hnsw_ef_search = cfg.index.hnsw_ef_search

    # Re-ranking parameter
    rerank_multiplier = cfg.rerank.multiplier

    try:
        # Initialize embedding model
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=cfg.embedding_model.name,
            cache_folder=cfg.hf_cache_dir,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        logging.info(f"Initialized embedding model: {cfg.embedding_model.name}")
    except Exception as e:
        logging.error(f"Error initializing embedding model: {e}")
        raise

    try:
        # Initialize Cross-Encoder model
        cross_encoder_model = CrossEncoder(
            model_name=cfg.cross_encoder.name,
            cache_dir=cfg.hf_cache_dir,
            device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        logging.info(f"Initialized Re-ranker: {cfg.cross_encoder.name}")
    except Exception as e:
        logging.error(f"Error initializing Cross-Encoder model: {e}")
        raise

    logging.info(f"Generating embedding for query: '{query_text}'")
    
    with torch.no_grad():
        query_embedding = Settings.embed_model.get_query_embedding(query_text)

    conn = None
    initial_retrieval_res = []
    try:
        conn = get_db_connection(cfg)
        cur = conn.cursor()

        # Set 'ef_search' parameter for HNSW
        cur.execute(f"SET hnsw.ef_search = {hnsw_ef_search};")
        conn.commit()

        logging.info(f"Set hnsw.ef_search = {hnsw_ef_search}")

        # Initial Vector Similarity Search
        initial_limit = top_k * rerank_multiplier
        logging.info(f"Executing initial vector search query (limit={initial_limit})")

        search_sql = f"""
        SELECT node_id, text, metadata
        FROM {table_name}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """

        cur.execute(search_sql, (query_embedding, initial_limit))
        initial_results = cur.fetchall()
        logging.info(f"Retrieved {len(initial_results)} initial candidates")

        # Format results
        for row in initial_results:
            node_id, text, metadata = row
            try:
                metadata_dict = (
                    json.loads(metadata)
                    if isinstance(metadata, str)
                    else metadata
                )
            except json.JSONDecodeError:
                logging.warning(f"Could not decode metadata for node {node_id}. Storing as raw.")
                metadata_dict = metadata

            initial_retrieval_res.append({
                "node_id": node_id,
                "text": text,
                "metadata": metadata_dict
            })

        cur.close()
        conn.close()
        logging.info("Database connection closed after retrieval.")

    except psycopg2.Error as e:
        logging.error(f"Database error during retrieval: {e}")
        if conn:
            try:
                conn.close()
            except Exception:
                pass
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred during retrieval: {e}")
        if conn:
            try:
                conn.close()
            except Exception:
                pass
        raise

    if not initial_retrieval_res:
        logging.info('⚠️ No initial candidates retrieved, skipping re-ranking')
        return []

    logging.info(f"Starting re-ranking process on {len(initial_retrieval_res)} candidates")
    rerank_start_time = time.time()

    sent_pairs = [(query_text, doc['text']) for doc in initial_retrieval_res]
    
    with torch.no_grad():
        rerank_scores = cross_encoder_model.predict(sent_pairs)

    scored_docs = []
    for i, doc in enumerate(initial_retrieval_res):
        doc["score"] = float(rerank_scores[i])
        scored_docs.append(doc)

    reranked_docs = sorted(scored_docs, key=lambda x: x['score'], reverse=True)

    rerank_end_time = time.time()
    rerank_duration = rerank_end_time - rerank_start_time
    logging.info(f"Re-ranking completed in {rerank_duration:.2f} sec")

    final_relevant_docs = reranked_docs[:top_k]
    logging.info(f"Returning top {len(final_relevant_docs)} docs after re-ranking")

    # Free up memory
    del sent_pairs, rerank_scores, initial_retrieval_res
    torch.cuda.empty_cache()
    gc.collect()

    return final_relevant_docs

@hydra.main(config_path='../conf/', config_name='config', version_base='1.3')
def main(cfg: DictConfig):
    test_query = input('Enter a query!:')
    top_k = int(input('Enter Top-K param:'))
    print(f"Searching for documents related to: '{test_query}'")

    try:
        relevant_docs = retrieve_relevant_docs(cfg, test_query, top_k=top_k)
        print("\n--- Retrieved Documents ---")
        if relevant_docs:
            for i, doc in enumerate(relevant_docs):
                print(f"--- Document {i+1} (Node ID: {doc['node_id']}, Score: {doc.get('score', 'N/A'):.4f}) ---")
                file_path = doc['metadata'].get('file_path', 'N/A')
                print(f"Source file: {file_path}")
                print(f"Metadata: {list(doc['metadata'].keys())}")
                print(f"Text: {doc['text']}")
                print("-" * 20)
        else:
            print("No relevant docs. found")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
