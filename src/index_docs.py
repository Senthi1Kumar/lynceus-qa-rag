"""
This script loads, processes, embeds, and indexes Markdown documents into 
a PostgreSQL database with pgvector support. It is configured using Hydra yaml
files located in the `conf/` directory and supports configurable 
embedding models, HNSW indexing, and markdown preprocessing options.
"""

import json
import os
import logging
import re
from typing import List
from tqdm import tqdm

import torch
from llama_index.readers.file import MarkdownReader
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

import psycopg2
import hydra
from omegaconf import DictConfig
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector


logging.basicConfig(level=logging.INFO)

def load_md_files(dir: str, remove_images: bool, remove_hyperlinks: bool):
    """Load markdown files from a directory."""
    md_reader = MarkdownReader(
        remove_images=remove_images,
        remove_hyperlinks=remove_hyperlinks,
    )

    # Get all markdown files in the directory
    files = [os.path.join(dir, f)
             for f in os.listdir(dir)
             if f.endswith('.md')]
    if not files:
        logging.warning(f"No markdown files found in directory: {dir}")
        return []
    logging.info(f"Found {len(files)} markdown files in directory: {dir}")

    # Load the markdown files
    docs = []
    for file in files:
        try:
            logging.info(f"Loading markdown file: {file}")
            docs.extend(md_reader.load_data(file))
        except Exception as e:
            logging.error(f"Error loading file {file}: {e}")
            continue
    if not docs:
        logging.warning("No documents were loaded from the markdown files.")
        return []
    logging.info(f"Loaded {len(docs)} documents from the markdown files")

    # Split the documents to chunks
    md_node_parser = MarkdownNodeParser()

    nodes = md_node_parser.get_nodes_from_documents(docs)

    logging.info(f"Parsed {len(nodes)} nodes/chunks from the documents")

    def postprocess_nodes_extract_images(
            nodes: List[TextNode]
            ) -> List[TextNode]:
        """
        Iterates through nodes, extracts image paths from text to metadata,
        and cleans the text content.
        """
        image_link_pattern = re.compile(r'!\[(.*?)\]\((.*?)\)')

        processed_nodes = []
        for node in nodes:
            original_text = node.get_content()
            found_images = image_link_pattern.findall(original_text)

            if found_images:
                node.metadata['image_paths'] = node.metadata.get(
                    'image_paths', []
                    )
                node.metadata['image_alt_texts'] = node.metadata.get(
                    'image_alt_texts', []
                    )

                for alt_text, image_path in found_images:
                    node.metadata['image_paths'].append(image_path.strip())
                    node.metadata['image_alt_texts'].append(alt_text.strip())

                cleaned_text = image_link_pattern.sub(
                    '', original_text
                    ).strip()
                cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text)
                node.set_content(cleaned_text)

            processed_nodes.append(node)

        return processed_nodes

    # Post-process nodes to extract images and clean text
    processed_nodes = postprocess_nodes_extract_images(nodes)

    logging.info(f"Processed {len(processed_nodes)} nodes to extract images and clean text")

    return processed_nodes


def index_md_files(cfg:DictConfig, nodes: List[TextNode]):
    """Embeds nodes and stores them directly in PostgreSQL."""
    logging.info('Connecting to Postgres for direct storage')

    # Set the Embedding model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=cfg.embedding_model.name,
        cache_folder=cfg.hf_cache_dir,
        device=device,
    )
    logging.info(f"Using embedding model: {cfg.embedding_model.name} on device: {device}")

    # DB connection params
    pg = cfg.db
    db_name = pg.db_name
    db_host = pg.db_host
    db_user = pg.db_user
    db_pwd = pg.db_pwd
    db_port = pg.db_port
    embed_dim = int(cfg.embedding_model.embed_dim)

    table_name = cfg.table.name
    index_name = f"{table_name}{cfg.index.name_suffix}"

    # HNSW index params 
    # (m=16, ef_construction=64, distance function=cosine distance)
    hnsw_m = cfg.index.hnsw_m
    hnsw_ef_construction = cfg.index.hnsw_ef_construction
    distance_func = cfg.index.distance_func

    if not all([db_name, db_host, db_user, db_pwd, embed_dim]):
        logging.error("Missing PostgreSQL connection or embedding dimension parameters in .env file.")
        raise ValueError("Database connection or embedding dimension parameters are not set in the environment variables.")

    conn = None
    try:
        conn = psycopg2.connect(
            dbname=db_name,
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_pwd
        )
        logging.info(f"Connected to PostgreSQL DB '{db_name}' on host '{db_host}'.")

        try:
            psycopg2.extras.register_default_json(conn)
            logging.info("Registered JSONB type for metadata.")
        except Exception as e:
            logging.warning(f"Could not register JSONB type: {e}")

        try:
            register_vector(conn)
            logging.info("Registered vector typecaster")
        except ImportError:
            logging.warning("pgvector extension not found. Ensure 'pgvector' is installed and available.")
        except Exception as e:
            logging.warning(f"Could not register vector type. Ensure 'pgvector' extension is enabled and accessible: {e}")

        cur = conn.cursor()

        # Create table if it doesn't exist
        # 1. `node_id`: unique identifier for the node (from LlamaIndex node.id_)
        # 2. `document_id`: original document ID (from node.metadata)
        # 3. `text`: the chunk content
        # 4. `metadata`: original node metadata (JSONB)
        # 5. `embedding`: the vector embedding (VECTOR type)
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            node_id UUID PRIMARY KEY,
            document_id TEXT,
            text TEXT,
            metadata JSONB,
            embedding VECTOR({embed_dim})
        );
        """
        cur.execute(create_table_sql)
        conn.commit()

        logging.info(f"Table '{table_name}' checked/created.")

        # Add HNSW index if it doesn't exist
        check_index_sql = """
        SELECT EXISTS (
            SELECT 1
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname = %s AND n.nspname = 'public'
        );
        """
        cur.execute(check_index_sql, (index_name,))
        index_exists = cur.fetchone()[0]

        if not index_exists:
            logging.info(f"Creating HNSW index '{index_name}' on table '{table_name}'...")
            create_index_sql = f"""
            CREATE INDEX {index_name} ON {table_name}
            USING hnsw (embedding {distance_func})
            WITH (m = {hnsw_m}, ef_construction = {hnsw_ef_construction});
            """

            try:
                cur.execute(create_index_sql)
                conn.commit()
                logging.info(f"HNSW index '{index_name}' created successfully")
            except psycopg2.errors.DuplicateObject:
                logging.info(f"HNSW index '{index_name}' already exists. Skipping creation.")
                conn.rollback()
            except psycopg2.Error as e:
                logging.error(f"Error creating HNSW index '{index_name}': {e}")
                conn.rollback()
                raise

        else:
            logging.info(f"HNSW index '{index_name}' already exists.")

        # Prepare data for batch insertion and generate embeddings
        data_to_insert = []
        logging.info(f"Generating embeddings and preparing data for insertion ({len(nodes)} nodes)...")

        embeddings = Settings.embed_model.get_text_embedding_batch(
            [node.get_content()
             for node in tqdm(nodes, desc="Generating Embeddings")]
            )

        for i, node in enumerate(nodes):
            document_id = node.metadata.get('document_id', node.id_)
            embedding = embeddings[i]

            data_to_insert.append((
                node.id_,
                document_id,
                node.get_content(),
                json.dumps(node.metadata),
                embedding
            ))

        logging.info("Data prepared for insertion.")

        insert_sql = f"""
        INSERT INTO {table_name} \
        (node_id, document_id, text, metadata, embedding)
        VALUES %s
        ON CONFLICT (node_id) DO NOTHING;
        """
        batch_size = 100
        logging.info(f"Starting batch insertion into '{table_name}' with batch size {batch_size}...")
        with tqdm(total=len(data_to_insert), desc="Inserting Batches") as pbar:
            for i in range(0, len(data_to_insert), batch_size):
                batch = data_to_insert[i:i + batch_size]
                try:
                    cur.execute("BEGIN;")
                    execute_values(cur, insert_sql, batch)
                    cur.execute("COMMIT;")
                    pbar.update(len(batch))
                except psycopg2.Error as db_err:
                    conn.rollback()
                    logging.error(f"Database error during batch insert {i // batch_size}: {db_err}")
                    raise
                except Exception as e:
                    conn.rollback()
                    logging.error(f"An unexpected error occurred during batch insert {i // batch_size}: {e}")
                    raise

        logging.info("Batch insertion completed.")

        cur.close()
        conn.close()
        logging.info("Database connection closed.")
        return True

    except psycopg2.Error as e:
        logging.error(f"Database connection or operation error: {e}")
        if conn:
            try:
                conn.close()
            except Exception:
                pass
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        if conn:
            try:
                conn.close()
            except Exception:
                pass
        raise

@hydra.main(config_path='../conf/', config_name='config', version_base='1.3')
def main(cfg: DictConfig):
    nodes = load_md_files(
        cfg.directory,
        remove_images=cfg.remove_images,
        remove_hyperlinks=cfg.remove_hyperlinks
    )
    if nodes:
        try:
            logging.info('Starting indexing process...')
            index_md_files(cfg=cfg, nodes=nodes)
            logging.info("✅ Indexing completed successfully and data saved to PostgreSQL")
        except Exception as e:
            logging.error(f"Error during direct indexing process: {e}")
    else:
        logging.warning("No nodes were processed.")


if __name__ == "__main__":
    main()
