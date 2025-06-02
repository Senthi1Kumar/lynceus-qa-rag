import os
import logging
import time
import json
from dotenv import load_dotenv

import mlflow
import streamlit as st

from llama_index.core import Settings
from llama_index.llms.google_genai import GoogleGenAI
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize, compose
from omegaconf import DictConfig

try:
    from src.retrieve_docs import get_db_connection, retrieve_relevant_docs
except ImportError:
    st.error("Could not import `retrieve_docs`. Make sure it's in the 'src' directory.")
    st.stop()


# Logging
logging.basicConfig(level=logging.INFO)

if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

# Hydra config
initialize(config_path='conf/', job_name='app', version_base='1.3')
cfg: DictConfig = compose(config_name='config')

# MLflow setup
try:
    mlflow.set_experiment('Lynceus Assistant')
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    # Trace and log the chats
    mlflow.llama_index.autolog()
    logging.info("MLflow tracking initialized.")
except Exception as e:
    logging.warning(f"MLflow initialization failed: {e}. MLflow logging will be skipped.")
    st.session_state['mlflow_available'] = False

if 'mlflow_available' not in st.session_state:
    st.session_state['mlflow_available'] = True


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize LLM
@st.cache_resource
def initialize_models():
    """Initializes and caches the LLM."""
    logging.info("Initializing Gemini LLM...")
    llm_instance = None

    try:
        if not GOOGLE_API_KEY:
            st.error("Google API Key not found in environment variables. Cannot initialize Gemini.")
            logging.error("GOOGLE_API_KEY not found for Gemini initialization.")
            return None

        llm_instance = GoogleGenAI(
            model=cfg.gemini.model,
            api_key=GOOGLE_API_KEY,
            max_tokens=4096,
            # temperature=0.1,
        )
        logging.info(f"Initialized Gemini LLM: {cfg.gemini.model}")

        # Basic check
        if llm_instance:
            try:
                test_prompt = "Hello"
                response = llm_instance.complete(test_prompt)
                logging.info(f"LLM test call successful for Gemini {cfg.gemini.model}. Response: {response.text[:50]}...")
                Settings.llm = llm_instance
                return llm_instance
            except Exception as llm_e:
                logging.error(f"LLM test call failed for Gemini {cfg.gemini.model}: {llm_e}")
                st.error(f"Could not connect to Gemini model '{cfg.gemini.model}'. Please ensure it is available and accessible.")
                return None
        else:
            return None

    except Exception as e:
        logging.error(f"Failed to initialize Gemini LLM: {e}")
        st.error(f"Failed to initialize LLM: {e}")
        return None

def check_db_status():
    """Checks if the database is reachable."""
    try:
        conn = get_db_connection(cfg)
        conn.close()
        return True
    except Exception:
        return False

def process_rag_query(query: str, llm: GoogleGenAI, top_k: int, mlflow_active: bool):
    """
    Processes a single RAG query: retrieves docs, generates answer, logs to MLflow.
    Returns the assistant's response content (answer, sources, images).
    """
    relevant_docs = []
    full_response_text = ""
    all_img_paths_from_src = []
    context_docs_metadata_for_logging = []
    mlflow_status = "Failed"

    try:
        # 1. Retrieve and Re-rank relevant documents
        retrieval_start_time = time.time()
        relevant_docs = retrieve_relevant_docs(cfg, query, top_k=top_k)
        retrieval_end_time = time.time()
        retrieval_duration = retrieval_end_time - retrieval_start_time

        if mlflow_active:
            mlflow.log_metric("retrieval_and_rerank_duration_seconds", retrieval_duration)
            mlflow.log_metric("num_final_retrieved_docs", len(relevant_docs))


        if not relevant_docs:
            full_response_text = "No relevant documents found for your query."
            mlflow_status = "No relevant documents found"
        else:
            # 2. Prepare context for the LLM
            context = "\n\n---\n\n".join([doc['text'] for doc in relevant_docs])

            # Collect images and metadata for logging
            for doc in relevant_docs:
                if isinstance(doc['metadata'], dict):
                    img_paths_in_chunk = doc['metadata'].get('image_paths', [])
                    if isinstance(img_paths_in_chunk, list):
                        base64_images = [p for p in img_paths_in_chunk if isinstance(p, str) and p.startswith("data:image/")]
                        all_img_paths_from_src.extend(base64_images)
                context_docs_metadata_for_logging.append(doc['metadata'])

            # Log context details
            if mlflow_active:
                mlflow.log_param("context_length_chars", len(context))
                if context_docs_metadata_for_logging:
                    try:
                        mlflow.log_text(json.dumps(context_docs_metadata_for_logging, indent=2), "context_docs_metadata.json")
                    except Exception as json_log_e:
                        logging.warning(f"Could not log context metadata as JSON artifact: {json_log_e}")
                        mlflow.log_text(str(context_docs_metadata_for_logging), "context_docs_metadata_str.txt")


            if not context:
                full_response_text = "Retrieved documents were empty or too short to form context. Could not generate an answer."
                mlflow_status = "Context empty after retrieval"
            else:
                # 3. Generate answer using the LLM
                prompt = f"""You are an AI assistant specialized in semiconductor manufacturing and documentation.
                Answer the following question based only on the provided context.
                If you cannot answer the question based on the context, politely state that the information is not available in the provided documents.

                Context:
                {context}

                Question: {query}

                Answer:
                """
                logging.info(f"Sending prompt to LLM w/ context length: {len(context)} characters...")

                llm_start_time = time.time()
                response = llm.complete(prompt)
                full_response_text = response.text
                llm_end_time = time.time()
                llm_duration = llm_end_time - llm_start_time
                logging.info(f"LLM complete call completed in {llm_duration:.2f} seconds.")

                if mlflow_active:
                    mlflow.log_metric("llm_generation_duration_seconds", llm_duration)
                    mlflow.log_param("streaming_supported", False)
                    mlflow.log_text(full_response_text, "full_llm_answer.txt")
                    mlflow.log_text(prompt, 'full_prompt.txt')
                    mlflow_status = "Success"

    except Exception as e:
        logging.error(f"An error occurred during RAG process: {e}")
        full_response_text = f"An error occurred during processing: {e}"
        mlflow_status = "Error during RAG process"
        if mlflow_active:
            mlflow.log_text(str(e), 'error_message.txt')

    finally:
        if mlflow_active:
            mlflow.log_param('status', mlflow_status)

    return {
        "answer": full_response_text,
        "sources": relevant_docs,
        "images": all_img_paths_from_src
    }


# App Layout
st.title("Lynceus Assistant")

if not check_db_status():
    st.error("Could not connect to the PostgreSQL DB. Please ensure the DB is running and connection parameters are correct.", icon="🚨")
    st.stop()

llm = initialize_models()

if llm is None:
    st.warning("Please check your 'config' and ensure the selected model/provider is available.")
    st.stop()


st.sidebar.header("Retrieval Settings")
top_k = st.sidebar.slider("Number of relevant documents to retrieve (after re-ranking):", 1, 10, cfg.retrieve.top_k)


st.write("Ask questions about the semiconductor documentation.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            if "images" in message and message["images"]:
                st.subheader("Images from Sources:")
                displayed_imgs = set()
                for img_path in message["images"]:
                    if isinstance(img_path, str) and img_path.startswith("data:image/") and img_path not in displayed_imgs:
                        try:
                            st.image(img_path)
                            displayed_imgs.add(img_path)
                        except Exception as img_e:
                            logging.warning(f"Could not display image from path {img_path[:50]}... : {img_e}")
                            st.warning('Could not display an image from the source.')

            if "sources" in message and message["sources"]:
                st.subheader("Sources Used:")
                for i, doc in enumerate(message["sources"]):
                    metadata_display = doc['metadata'].get('file_path', 'N/A')
                    rerank_score_display = f", Re-rank Score: {doc.get('score', 'N/A'):.4f}" if "score" in doc and doc.get('score') is not None else ""
                    st.write(f"**Source {i+1}:** (Node ID: {doc['node_id']}, Source: {metadata_display}{rerank_score_display})")
                    original_chunk_txt = doc['text']
                    with st.expander(f"Show full original chunk {i+1}"):
                        st.text(original_chunk_txt)


# User input
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    mlflow_active = st.session_state.get('mlflow_available', True)
    if mlflow_active:
        mlflow_run_context = mlflow.start_run(run_name=f"Query: {prompt[:50]}...",
                                              log_system_metrics=True)
        mlflow_run_context.__enter__()
        mlflow.log_param("query", prompt)
        mlflow.log_param("top_k_retrieval", top_k)
        mlflow.log_param("llm_provider", "Gemini")
        mlflow.log_param("llm_model", cfg.gemini.model)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                assistant_response = process_rag_query(prompt, llm, top_k, mlflow_active)

                st.markdown(assistant_response["answer"])

                if assistant_response["images"]:
                    st.subheader("Images from Sources:")
                    displayed_imgs = set()
                    for img_path in assistant_response["images"]:
                        if isinstance(img_path, str) and img_path.startswith("data:image/") and img_path not in displayed_imgs:
                            try:
                                st.image(img_path)
                                displayed_imgs.add(img_path)
                            except Exception as img_e:
                                logging.warning(f"Could not display image from path {img_path[:50]}... : {img_e}")
                                st.warning('Could not display an image from the source.')

                if assistant_response["sources"]:
                    st.subheader("Sources Used:")
                    for i, doc in enumerate(assistant_response["sources"]):
                        metadata_display = doc['metadata'].get('file_path', 'N/A')
                        rerank_score_display = f", Re-rank Score: {doc.get('score', 'N/A'):.4f}" if "score" in doc and doc.get('score') is not None else ""
                        st.write(f"**Source {i+1}:** (Node ID: {doc['node_id']}, Source: {metadata_display}{rerank_score_display})")
                        original_chunk_txt = doc['text']
                        with st.expander(f"Show full original chunk {i+1}"):
                            st.text(original_chunk_txt)


            except Exception as e:
                logging.error(f"An error occurred during RAG process: {e}")
                st.error(f"An error occurred: {e}")
                if mlflow_active:
                    mlflow.log_param('status', 'Error')
                    mlflow.log_text(str(e), 'error_message.txt')
                st.markdown(f"Error: {e}")
            finally:
                if mlflow_active:
                    mlflow_run_context.__exit__(None, None, None)

    st.session_state.messages.append({
        "role": "assistant",
        "content": assistant_response["answer"],
        "sources": assistant_response["sources"],
        "images": assistant_response["images"]
    })
