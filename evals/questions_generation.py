"""
This script generates synthetic questions using gemini's function-calling for 
evaluating retrieval and reranking components.
"""

import os
import json
import logging
import time
from typing import List, Dict, Union, Optional
from hydra import initialize, compose
from omegaconf import DictConfig

import pandas as pd
from google import genai
from google.genai import types
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# Hydra config
initialize(config_path='../conf/', job_name='app', version_base='1.3')
cfg: DictConfig = compose(config_name='config')

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logging.error("GOOGLE_API_KEY not found in .env file.")
    raise ValueError("GOOGLE_API_KEY is required.")


CSV_FILE_PATH = cfg.eval.csv_file
NODE_ID_COLUMN = "node_id"
TEXT_COLUMN = "text"

# Node IDs for question generation
## NOTE: I dumped the table as CSV from Postgres DB, and manually selected node ids 
## based on the text content w.r.t manuals/docs. So, if you want to evaluate it,
## change the node ids in `context_ids` according your indexed nodes in your DB.

context_ids = {
    'q1': '94a73444-bcd4-4814-8577-5e4e6db87e27',
    'q2': '3fcb06f4-69c5-4cc9-a3c3-0ad67a507dc5',
    'q3': '03a87c7f-7ba1-4182-838b-9869bb94cb4d',
    'q4': '553f9bc8-eda2-4712-8b2d-b0a4df19f7f6',
    'q5': '50aac2b4-8cc5-44c3-96f8-b5eef9c70c96',
    'q6': ['f844eb72-8c07-486a-9fd7-d7e25412f55c', '280d1ce0-cef7-46d5-8034-86f57bd16a1e', '08bfd448-376a-482f-94b0-91777dc55668'],
    'q7': ['7346f4bd-fd1c-4db5-8333-f645760843ed', 'f7a28b42-f50d-4902-8a25-68a26aed5387', 'bc85f290-330d-4a1d-9c8d-baa62f0bb999'],
    'q8': '39cc1fd9-8cc1-468f-9b7b-d82d15638d39',
    'q9': ['04e8454e-8e7e-463a-b693-51116f0e22a2', '9a8fc055-d51b-4302-9403-b772640cc635'],
    'q10': '43bf46f7-bfed-4096-8879-76e1b9c67e78',
    'q11': ['71762d1a-2df3-457f-a413-ec324f6059b5', 'aa91d5b1-91ca-4bd3-9da7-fd1c33a0764e', '8ca943f3-ab8e-4b1d-a2fa-b411984861b5'],
    'q12': '71544e8a-ed0d-437a-a0be-87b6e9f128e6',
    'q13': 'd4d15997-6896-4fff-85ee-785c69108eb8',
    'q14': ['7ca30d1d-d5ec-40ef-819a-25903c18b257', '5cdabdc0-7893-41cf-8620-f750f149474d', '8b43475e-1f52-4f23-8656-febb8af755d1', '105cdf15-fceb-480c-8102-61c4a3dfe894', '76ec91c0-ae71-46d3-81ed-a8d5e8922f64'],
    'q15': 'aa78563e-5c4b-4d2c-8fc2-9a14db975730',
    'q16': '3474b32b-9e0a-4ed3-803e-92d82d374409',
    'q17': ['5dbb1f55-c7d3-4f4f-8fa4-e42a4459544c', '047f4cf3-d30c-4c05-8bea-4aa3eb0aee11'],
    'q18': '1aa1aeee-61a4-46a5-9312-8c32d884c665',
    'q19': ['abfa4d7a-e73f-4028-a316-29e044f88a6a', 'c52ac521-cd54-4319-9086-8c084742a582'],
    'q20': ['d4941d6b-0423-42c1-9441-7a0bae6a4a63', '7a8db9d5-4f38-401d-9c8b-743ee5e86e24', '6c4c4a33-ac9f-4a67-bb1f-882e99939d6c', '104f6909-0147-49cf-8025-3d73226c0277', 'aa867c68-d820-4e19-acdf-56cd58879293'],
    'q21': 'c816f374-4f74-4202-8c28-62c35e566fea',
    'q22': ['e0928a6e-583d-4d3d-a3cd-3f3ea93bf1e9', 'b212da87-083c-4ba4-8874-bd79c1124614', '034e200d-89ce-4ca9-8e3a-5b2ab640937f'],
    'q23': '18552086-5590-4dbd-83aa-fc25bd7eaf95',
    'q24': '17fddfa2-2303-4f08-a5ee-cab7fc0a6f82',
    'q25': ['90c01c44-8182-4b7c-8aac-8676bd27fbf4', '3d4f730a-8c22-436f-8fef-8880802b0bf8', '062139b0-0a6b-40e1-be91-0b47d2de5f2b']
}


def load_nodes_dataframe(csv_path: str) -> Optional[pd.DataFrame]:
    """Loads the CSV file containing node data into a pandas DataFrame."""
    if not os.path.exists(csv_path):
        logging.error(f"CSV file not found at path: {csv_path}")
        return None
    try:
        df = pd.read_csv(csv_path)
        if NODE_ID_COLUMN not in df.columns or TEXT_COLUMN not in df.columns:
            logging.error(
                f"CSV file must contain columns named '{NODE_ID_COLUMN}' and '{TEXT_COLUMN}'. "
                f"Found columns: {df.columns.tolist()}"
            )
            return None
        logging.info(f"Successfully loaded CSV from {csv_path} with {len(df)} rows.")
        return df
    except Exception as e:
        logging.error(f"Error loading CSV file {csv_path}: {e}")
        return None


nodes_df: Optional[pd.DataFrame] = None

def get_node_content_from_dataframe(node_ids: Union[str, List[str]]) -> Optional[str]:
    """
    Fetches and concatenates the text content of specified node_ids from the loaded DataFrame.
    """
    global nodes_df
    if nodes_df is None:
        logging.error("Nodes DataFrame not loaded. Call load_nodes_dataframe() first or ensure CSV_FILE_PATH is correct.")
        return None

    if not node_ids:
        return None

    if isinstance(node_ids, str):
        node_ids_list = [node_ids]
    else:
        node_ids_list = node_ids
    
    try:
        # Filter the DataFrame for the given node_ids
        # Ensure `node_id` column in DataFrame is of the same type as node_ids_list elements
        # Assuming node_ids in CSV are strings:
        matching_rows = nodes_df[nodes_df[NODE_ID_COLUMN].astype(str).isin(node_ids_list)]
        
        if matching_rows.empty:
            logging.warning(f"No matching rows found in DataFrame for node_ids: {node_ids_list}")
            return None
            
        all_text_content = matching_rows[TEXT_COLUMN].tolist()
        logging.info(f"Fetched content for {len(all_text_content)} out of {len(node_ids_list)} node_ids from DataFrame.")
        return "\n\n---\n\n".join(str(text) for text in all_text_content) if all_text_content else None

    except Exception as e:
        logging.error(f"Error fetching node content from DataFrame: {e}")
        return None

# Define the function declaration for the model
generate_qn_func = {
    "name": "generate_troubleshooting_question",
    "description": (
        "Generates a specific troubleshooting question based on the provided context. "
        "The question should be phrased as if a user is experiencing a problem related "
        "to semiconductor devices or manufacturing and needs help. "
        "If multiple context pieces are provided, the question should ideally require "
        "understanding or synthesis of information across these pieces."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "generated_question": {
                "type": "string",
                "description": "The troubleshooting question generated from the context. Should be brief and concise.",
            }
        },
        "required": ["generated_question"]
    }
}

# Client config and tools
client = genai.Client(api_key=GOOGLE_API_KEY)
tools = types.Tool(function_declarations=[generate_qn_func])
config = types.GenerateContentConfig(tools=[tools], temperature=0.6)


def generate_synthetic_questions(context_node_ids: Dict[str, Union[str, List[str]]]) -> List[Dict]:
    """
    Generates synthetic troubleshooting questions for each context provided,
    fetching context from the pre-loaded DataFrame.
    """
    global nodes_df
    if nodes_df is None:
        logging.error("Nodes DataFrame is not loaded. Cannot generate questions.")
        return [{"query_id": q_key, "error": "Nodes DataFrame not loaded"} for q_key in context_node_ids.keys()]

    generated_eval_set = []

    for q_key, node_ids in context_node_ids.items():
        logging.info(f"Processing {q_key} with node_id(s): {node_ids}")
        
        context_text = get_node_content_from_dataframe(node_ids)

        if not context_text:
            logging.warning(f"Could not retrieve context for {q_key} from DataFrame. Skipping.")
            generated_eval_set.append({
                "query_id": q_key,
                "reference_node_ids": [node_ids] if isinstance(node_ids, str) else node_ids,
                "context_text_retrieved": False,
                "generated_question": None,
                "error": "Context not found in DataFrame"
            })
            continue
        
        num_chunks_provided = 1 if isinstance(node_ids, str) else len(node_ids)

        prompt = (
            f"You are an AI assistant tasked with creating high-quality, specific evaluation questions for a troubleshooting system "
            f"in the semiconductor domain. Your goal is to generate questions that are **brief, concise (ideally one sentence, under 25 words), "
            f"and directly answerable ONLY from the provided context.**\n\n"
            f"CONTEXT (consisting of {num_chunks_provided} chunk(s) of technical documentation):\n"
            f"```\n{context_text}\n```\n\n"
            f"INSTRUCTIONS:\n"
            f"1.  Analyze the core information in the context.\n"
            f"2.  If the context describes a problem/solution, formulate a question about identifying or resolving that specific problem.\n"
            f"3.  If the context describes a process, ask a troubleshooting question about a potential failure, anomaly, or critical parameter within that process.\n"
        )
        if num_chunks_provided > 1:
            prompt += (
                f"4.  **IMPORTANT (Multi-Context Synthesis):** Since {num_chunks_provided} context chunks are provided, your question MUST "
                f"require understanding and synthesizing information from AT LEAST TWO of these different chunks to formulate the question and its answer. "
                f"Do not generate a question that can be answered from a single chunk if multiple are given.\n"
                f"5.  The question should be phrased as a real user troubleshooting an issue.\n"
            )
        else:
            prompt += (
                "4.  The question should be phrased as a real user troubleshooting an issue.\n"
            )
        prompt += (
            "6.  Ensure the question is distinct and focuses on a key technical detail from the context.\n"
            "7.  Avoid ambiguity. The question should have a clear, targeted answer within the context.\n\n"
            "Now, use the 'generate_troubleshooting_question' tool to formulate this specific question."
        )

        try:
            time.sleep(7)
            response = client.models.generate_content(
                model=cfg.gemini.qn_generator_model,
                contents=prompt,
                config=config
            )
            
            question_generated = None
            error_message = None

            if response.candidates and response.candidates[0].content.parts:
                function_call_part = None
                for part in response.candidates[0].content.parts:
                    if part.function_call:
                        function_call_part = part.function_call
                        break
                
                if function_call_part and function_call_part.name == "generate_troubleshooting_question":
                    args = function_call_part.args
                    question_generated = args.get("generated_question")
                    if question_generated:
                         logging.info(f"Successfully generated question for {q_key}: {question_generated}")
                    else:
                        logging.warning(f"Tool called for {q_key}, but 'generated_question' was missing or empty in args: {args}")
                        error_message = "Tool called, but 'generated_question' missing/empty"
                else:
                    logging.warning(f"No valid function call 'generate_troubleshooting_question' found for {q_key}. LLM response: {response.text if hasattr(response, 'text') else 'No text response'}")
                    error_message = "No valid function call in LLM response"
            else:
                logging.warning(f"No response candidates or parts found for {q_key}.")
                error_message = "No response candidates/parts from LLM"

            generated_eval_set.append({
                "query_id": q_key,
                "reference_node_ids": [node_ids] if isinstance(node_ids, str) else node_ids,
                "context_text_retrieved": True,
                "generated_question": question_generated,
                "error": error_message
            })

        except Exception as e:
            logging.error(f"Error calling Gemini API for {q_key}: {e}")
            generated_eval_set.append({
                "query_id": q_key,
                "reference_node_ids": [node_ids] if isinstance(node_ids, str) else node_ids,
                "context_text_retrieved": True, 
                "generated_question": None,
                "error": str(e)
            })
        
    return generated_eval_set


if __name__ == "__main__":
    nodes_df = load_nodes_dataframe(CSV_FILE_PATH)

    if nodes_df is not None:
        logging.info("Starting synthetic question generation using CSV data...")
        synthetic_questions_data = generate_synthetic_questions(context_ids)
        logging.info("Synthetic question generation complete.")

        output_filename = "synthetic_evaluation_questions.json"
        with open(output_filename, "w") as f:
            json.dump(synthetic_questions_data, f, indent=4)
        logging.info(f"Generated questions saved to {output_filename}")

        successful_generations = sum(1 for item in synthetic_questions_data if item["generated_question"])
        failed_generations = len(synthetic_questions_data) - successful_generations
        logging.info(f"Successfully generated {successful_generations} questions.")
        logging.info(f"Failed to generate {failed_generations} questions (check logs and output file for details).")
    else:
        logging.error(f"Could not load data from CSV: {CSV_FILE_PATH}. Question generation aborted.")

