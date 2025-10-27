# backend_utils.py (Revised with Pre-classification & Validation Logging, RAG 2 Restored)

import pandas as pd
import numpy as np
import textwrap
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader # Removed ExcelLoader, handled by pandas
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_openai import OpenAIEmbeddings # Option if using OpenAI embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.schema import BaseRetriever
import json
import time
from typing import List, Dict, Optional, Any, Tuple
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
import os
import asyncio
from sklearn.metrics.pairwise import cosine_similarity
import logging # Added for validation logging

# Get the directory of the currently executing file
BACKEND_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- VALIDATION LOGGING SETUP ---
def setup_validation_logger():
    """Sets up a logger to record LLM outputs and similarity scores."""
    logger = logging.getLogger('validation_logger')
    # Prevent duplicate handlers if function is called multiple times
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        # Use absolute path for the log file
        log_file_path = os.path.join(BACKEND_SCRIPT_DIR, 'validation_log.jsonl')
        handler = logging.FileHandler(log_file_path, mode='w') # 'w' overwrites log each run
        # JSON Lines format: each log record is a valid JSON object on a new line
        formatter = logging.Formatter('{"timestamp": "%(asctime)s", "level": "%(levelname)s", "data": %(message)s}')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        print(f"📝 Validation logger initialized. Logging to: {log_file_path}")
    return logger

validation_logger = setup_validation_logger()

# --- EMBEDDING MODEL CONFIGURATION ---
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
K = 2 # Number of top results to retrieve per retriever
try:
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    print("✅ Embeddings model loaded successfully.")
except Exception as e:
    print(f"❌ CRITICAL ERROR: Failed to load embeddings model: {e}")
    embeddings = None

# --- LLM CONFIGURATION for OpenAI ---
LLM_MODEL = "gpt-4o-mini"
MAX_RETRIES = 3
load_dotenv()
api_key_from_env = os.getenv("OPENAI_API_KEY")

client: Optional[OpenAI] = None
async_client: Optional[AsyncOpenAI] = None

if api_key_from_env:
    try:
        client = OpenAI(api_key=api_key_from_env)
        async_client = AsyncOpenAI(api_key=api_key_from_env)
        print("✅ OpenAI clients (sync & async) initialized successfully.")
    except Exception as e:
        print(f"❌ FATAL ERROR: Failed to initialize OpenAI clients: {e}")
else:
    print("⚠️ OPENAI_API_KEY environment variable is NOT set. LLM functions will fail.")

# --- CHECKLIST DEFINITION ---
checklist = [
    {"item": "HDB Approval & Compliance", "question": "Does the agreement mention HDB approval, compliance with HDB rules, and require valid immigration passes (min. 6 months) for all occupiers?"},
    {"item": "Lease Term", "question": "Are the exact start date, end date, and duration (minimum 6 months for HDB) clearly stated?"},
    {"item": "Rent Payment", "question": "Is the monthly rent amount, payment due date, and payment method clearly specified?"},
    {"item": "Security Deposit", "question": "What are the terms for the security deposit amount, refund period (standard is 14 days post-lease), conditions for deductions (requiring 14 days notice to tenant to fix), and is using it for the last month's rent prohibited?"},
    {"item": "Problem-Free Period", "question": "Is a 'Problem-Free Period' (usually 30 days) defined at the start, stating the Landlord is responsible for fixing any defects reported by the tenant during this time?"},
    {"item": "Minor Repairs", "question": "Is a specific dollar amount threshold (e.g., $150-$200) defined? Does it state the tenant pays up to this threshold per incident, and the landlord pays the excess unless the tenant caused the damage?"},
    {"item": "Air-Con Servicing", "question": "Does the agreement require the tenant to arrange and pay for regular air-con servicing (e.g., quarterly) and keep receipts as proof?"},
    {"item": "Landlord Access", "question": "Does the clause specify a required notice period (standard is 48 hours written notice) for landlord entry for non-emergency inspections, repairs, or viewings?"},
    {"item": "Quiet Enjoyment", "question": "Is the tenant's right to 'peaceably hold and enjoy' the property without unreasonable interruption from the landlord explicitly guaranteed?"},
    {"item": "Subletting", "question": "Does the agreement explicitly prohibit assigning, subletting, or parting with possession of the flat without the landlord's written consent and HDB approval?"},
    {"item": "Pets", "question": "Does the agreement address pets, requiring landlord consent and acknowledging HDB rules?"},
    {"item": "Smoking", "question": "Is smoking explicitly prohibited inside the HDB flat?"},
    {"item": "Diplomatic / Break Clause", "question": "If a diplomatic or break clause exists (check Schedule), does it clearly define the conditions (job loss, transfer), minimum stay period, notice period, and any compensation required for early termination?"},
    {"item": "Inventory List", "question": "Does the agreement mandate the creation and signing of an Inventory List prepared by the landlord at the start of the lease?"},
    {"item": "Property Condition Report", "question": "Does the agreement mandate the creation and signing of a Property Condition Report prepared by the tenant within the Problem-Free Period?"},
    {"item": "End of Tenancy Obligations", "question": "Does the agreement specify tenant responsibilities upon moving out, such as returning the flat to its original condition (fair wear and tear excepted), professional curtain cleaning, returning keys, and NOT terminating utilities before handover?"},
]

# --- RAG 1: Ideal Clauses (Annotated CEA Template) ---
IDEAL_CLAUSES_DATA_DIR = os.path.join(BACKEND_SCRIPT_DIR, "TA_template")
IDEAL_CLAUSES_FAISS = os.path.join(BACKEND_SCRIPT_DIR, "faiss_index_ideal_clauses")
ideal_clauses_retriever: Optional[BaseRetriever] = None

def build_or_load_ideal_clauses_retriever(rebuild_index: bool = False) -> Optional[BaseRetriever]:
    """ Builds or loads RAG 1 index. Returns Retriever or None. """
    global embeddings
    if embeddings is None: return None

    # ... (Keep the rest of this function exactly as in the previous version) ...
    # It correctly loads/builds the FAISS index for the annotated CEA template.
    if rebuild_index or not os.path.exists(IDEAL_CLAUSES_FAISS):
        print("\n--- BUILDING/REBUILDING RAG 1: IDEAL CLAUSES RETRIEVER ---")
        if not os.path.exists(IDEAL_CLAUSES_DATA_DIR):
             print(f"❌ Error: RAG 1 Data directory not found at {IDEAL_CLAUSES_DATA_DIR}")
             return None
        try:
            loader = DirectoryLoader(
                path=IDEAL_CLAUSES_DATA_DIR, glob="**/*.pdf", # Or .md if using annotated markdown
                loader_cls=PyPDFLoader, # Change if using markdown loader
                show_progress=True
            )
            all_documents = loader.load()
            print(f"Loaded {len(all_documents)} source document pages/files for RAG 1.")

            custom_separators = ["\n\n", r"\n\s*[A-Z]+\s+\d*\s*\.", r"\n\s*\d+\.\d*\s*", r"\n\s*\([a-zA-Z0-9]+\)\s*", "\n", " ", ""]
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200,
                separators=custom_separators, is_separator_regex=True
            )
            all_chunks = text_splitter.split_documents(all_documents)
            print(f"Split RAG 1 source into {len(all_chunks)} chunks.")
            if not all_chunks: return None

            vectorstore = FAISS.from_documents(all_chunks, embeddings)
            vectorstore.save_local(IDEAL_CLAUSES_FAISS)
            print(f"✅ RAG 1 FAISS index built and saved to {IDEAL_CLAUSES_FAISS}.")
            return vectorstore.as_retriever(search_kwargs={"k": K})
        except Exception as e:
            print(f"❌ Error building RAG 1 FAISS index: {e}")
            return None
    else:
        print("\n--- LOADING RAG 1: IDEAL CLAUSES RETRIEVER ---")
        try:
            vectorstore = FAISS.load_local(
                IDEAL_CLAUSES_FAISS, embeddings, allow_dangerous_deserialization=True
            )
            print(f"✅ RAG 1 FAISS index loaded successfully from {IDEAL_CLAUSES_FAISS}.")
            return vectorstore.as_retriever(search_kwargs={"k": K})
        except Exception as e:
            print(f"❌ Error loading RAG 1 FAISS index: {e}")
            return None

# --- RAG 2: General Q&A (Excel Source) --- RESTORED ---
GENERAL_QA_EXCEL_PATH = os.path.join(BACKEND_SCRIPT_DIR, "Database Requirements.xlsx") # Adjust path if needed
GENERAL_QA_FAISS = os.path.join(BACKEND_SCRIPT_DIR, "faiss_index_general_qa")
general_qa_retriever: Optional[BaseRetriever] = None

def build_or_load_general_qa_retriever(rebuild_index: bool = False) -> Optional[BaseRetriever]:
    """ Builds or loads RAG 2 index from Excel. Returns Retriever or None. """
    global embeddings
    if embeddings is None: return None
    # ... (Keep the rest of this function exactly as in your original script) ...
    # It correctly loads the Excel file, creates Documents, and builds/loads the RAG 2 FAISS index.
    if rebuild_index or not os.path.exists(GENERAL_QA_FAISS):
        print("\n--- BUILDING/REBUILDING RAG 2: GENERAL Q&A RETRIEVER ---")
        if not os.path.exists(GENERAL_QA_EXCEL_PATH):
            print(f"❌ Error: RAG 2 Excel file not found at {GENERAL_QA_EXCEL_PATH}")
            return None
        try:
            df = pd.read_excel(GENERAL_QA_EXCEL_PATH, header=1)
            print(f"Loaded {len(df)} rows from {GENERAL_QA_EXCEL_PATH}.")
            documents = []
            for index, row in df.iterrows():
                content = textwrap.dedent(f"""
                    Question: {row.get('Question', 'N/A')}
                    Answer/Explanation: {row.get('Answer / Explanation', 'N/A')}
                    Legal Context: {row.get('Legal Commentary', 'N/A')}
                    Regulation/Source: {row.get('Government Regulation / Explanation', 'N/A')}
                """).strip()
                doc = Document(page_content=content, metadata={"source": GENERAL_QA_EXCEL_PATH, "row_index": index})
                documents.append(doc)
            print(f"Created {len(documents)} General Q&A Documents.")
            if not documents: return None

            vectorstore = FAISS.from_documents(documents, embeddings)
            vectorstore.save_local(GENERAL_QA_FAISS)
            print(f"✅ RAG 2 FAISS index built and saved to {GENERAL_QA_FAISS}.")
            return vectorstore.as_retriever(search_kwargs={"k": K})
        except Exception as e:
            print(f"❌ Error building RAG 2 FAISS index: {e}")
            return None
    else:
        print("\n--- LOADING RAG 2: GENERAL Q&A RETRIEVER ---")
        try:
            vectorstore = FAISS.load_local(
                GENERAL_QA_FAISS, embeddings, allow_dangerous_deserialization=True
            )
            print(f"✅ RAG 2 FAISS index loaded successfully from {GENERAL_QA_FAISS}.")
            return vectorstore.as_retriever(search_kwargs={"k": K})
        except Exception as e:
            print(f"❌ Error loading RAG 2 FAISS index: {e}")
            return None


# --- User Document Processing ---
def load_and_extract_pdf_text(file_path: str) -> str:
    # (Keep function as previously defined)
    print(f"\n--- Loading and Extracting Text from: {os.path.basename(file_path)} ---")
    # ... (rest of the function) ...
    return full_text

def split_user_document(user_uploaded_text: str, source_name: str = "User TA") -> List[Document]:
    # (Keep function as previously defined)
    print(f"\n--- SPLITTING USER DOCUMENT: {source_name} ---")
    # ... (rest of the function) ...
    return user_chunks

# --- Pre-classification Function ---
def preclassify_chunks_by_similarity(
    user_chunks: List[Document],
    checklist: List[Dict[str, str]],
    embeddings_model: object,
    similarity_threshold: float = 0.70
) -> Dict[str, List[Document]]:
    # (Keep function as previously defined)
    print(f"\n--- Starting Pre-classification of {len(user_chunks)} User TA Chunks ---")
    # ... (rest of the function) ...
    return classification_map

# --- LLM Comparison Function (Asynchronous) ---
async def async_llm_compare_and_critique(
    user_clause_text: str,
    ideal_context_docs: List[Document],
    checklist_item_name: str,
    client_async: AsyncOpenAI
) -> Dict:
    # (Keep function as previously defined)
    # ... (rest of the function including schema, prompt, API call) ...
    return parsed_json # or error dict

# --- Main Analysis Function (Checklist-Guided, Parallelized, with Validation Logging) --- MODIFIED ---
async def run_checklist_analysis_async(
    classification_map: Dict[str, List[Document]],
    ideal_clauses_retriever: BaseRetriever,
    checklist: List[Dict[str, str]],
    embeddings_model: object # Pass embeddings model for validation
) -> List[Dict]:
    """
    Runs the checklist-guided RAG analysis asynchronously, calculates similarity
    for validation, and logs results.
    """
    global async_client, validation_logger
    if ideal_clauses_retriever is None or async_client is None or embeddings_model is None:
        raise RuntimeError("Analysis failed: RAG retriever, Async Client, or Embeddings model not initialized.")

    tasks = []
    print("\n🔍 Preparing parallel analysis tasks for checklist items...")

    analysis_inputs = [] # Store inputs for validation logging

    for item_dict in checklist:
        checklist_item_name = item_dict["item"]
        checklist_question_detail = item_dict["question"]

        # A) Retrieve standard context
        try:
            retrieved_standard_docs = ideal_clauses_retriever.invoke(checklist_item_name + ": " + checklist_question_detail)
        except Exception as e:
            print(f"⚠️ Error retrieving KB context for {checklist_item_name}: {e}")
            retrieved_standard_docs = []

        # B) Retrieve pre-classified user TA chunk(s)
        relevant_user_chunks = classification_map.get(checklist_item_name, [])
        user_ta_section_text = "\n---\n".join([chunk.page_content for chunk in relevant_user_chunks]) if relevant_user_chunks else ""

        # Store inputs needed for validation later
        analysis_inputs.append({
            "checklist_item": checklist_item_name,
            "user_chunks": relevant_user_chunks,
            "standard_docs": retrieved_standard_docs,
            "user_ta_section_text": user_ta_section_text # Text sent to LLM
        })

        # C) Create an async task for the LLM comparison
        tasks.append(
            async_llm_compare_and_critique(
                user_clause_text=user_ta_section_text,
                ideal_context_docs=retrieved_standard_docs,
                checklist_item_name=checklist_item_name,
                client_async=async_client
            )
        )

    # D) Execute all tasks concurrently
    print(f"🚀 Running {len(tasks)} analysis tasks in parallel...")
    start_time = time.time()
    results = await asyncio.gather(*tasks) # List of LLM output dicts
    end_time = time.time()
    print(f"✅ Parallel analysis complete in {end_time - start_time:.2f} seconds.")

    # E) Perform Validation Logging (after getting LLM results)
    print("📝 Performing validation logging...")
    validation_data_to_log = []
    for i, llm_result in enumerate(results):
        log_entry = {
            "checklist_item": analysis_inputs[i]["checklist_item"],
            "llm_output": llm_result,
            "max_similarity_score": None # Default to None
        }

        # Calculate Max Cosine Similarity for validation
        user_chunks_for_item = analysis_inputs[i]["user_chunks"]
        standard_docs_for_item = analysis_inputs[i]["standard_docs"]

        if user_chunks_for_item and standard_docs_for_item:
            try:
                user_texts = [chunk.page_content for chunk in user_chunks_for_item]
                standard_texts = [doc.page_content for doc in standard_docs_for_item]

                user_embeddings = embeddings_model.embed_documents(user_texts)
                standard_embeddings = embeddings_model.embed_documents(standard_texts)

                # Calculate all pairwise similarities
                similarity_matrix = cosine_similarity(user_embeddings, standard_embeddings)

                # Find the maximum similarity score between any user chunk and any standard chunk used
                max_similarity = np.max(similarity_matrix) if similarity_matrix.size > 0 else 0.0
                log_entry["max_similarity_score"] = float(max_similarity) # Ensure JSON serializable

            except Exception as e:
                print(f"⚠️ Error calculating similarity for '{log_entry['checklist_item']}': {e}")
                log_entry["similarity_error"] = str(e)

        # Log the combined data as a JSON string
        validation_logger.info(json.dumps(log_entry))
        validation_data_to_log.append(log_entry) # Also keep in memory if needed

    print("✅ Validation logging complete.")
    return results # Return only the LLM results for the main report

# Synchronous wrapper remains the same
def run_checklist_analysis(
    classification_map: Dict[str, List[Document]],
    ideal_clauses_retriever: BaseRetriever,
    checklist: List[Dict[str, str]]
) -> List[Dict]:
     global embeddings # Ensure embeddings model is accessible
     if embeddings is None: raise RuntimeError("Embeddings model not loaded for analysis.")
     # Pass embeddings model to the async function
     return asyncio.run(run_checklist_analysis_async(classification_map, ideal_clauses_retriever, checklist, embeddings))

# --- Report Formatting ---
def format_analysis_results_to_markdown(analysis_report_list: List[Dict]) -> str:
    # (Keep function as previously defined)
    md_parts = ["## 📋 Tenancy Agreement Analysis Report"] # Updated title slightly
    summary = {"Compliant": 0, "Partial": 0, "Missing": 0, "Unclear": 0, "High Risk": 0, "Errors": 0}

    for result in analysis_report_list:
        item_name = result.get("checklist_item", "Unknown Item")
        error = result.get("error")

        if error:
            md_parts.append(f"\n### {item_name}:")
            md_parts.append(f"🔴 **Analysis Error:** {error}")
            md_parts.append("---")
            summary["Errors"] += 1
            summary["High Risk"] += 1
            continue

        presence = result.get("presence", "Unclear")
        comparison = result.get("comparison_summary", "N/A")
        risk = result.get("risk_level", "N/A")
        suggestion = result.get("suggestion", "N/A")

        # Update summary counts
        if presence == "Yes": summary["Compliant"] += 1
        elif presence == "Partially": summary["Partial"] += 1
        elif presence == "No": summary["Missing"] += 1
        else: summary["Unclear"] +=1
        if risk == "HIGH": summary["High Risk"] +=1

        risk_color = "🟢" if risk == "LOW" else ("🟠" if risk == "MEDIUM" else ("🔴" if risk == "HIGH" else "⚪️"))
        presence_icon = "✅" if presence == "Yes" else ("⚠️" if presence == "Partially" else ("❌" if presence == "No" else "❓"))

        md_parts.append(f"\n### {presence_icon} {item_name}")
        md_parts.append(f"- **Presence:** {presence}")
        md_parts.append(f"- **Comparison:** {comparison}")
        md_parts.append(f"- **Risk Level:** {risk_color} **{risk}**")
        md_parts.append(f"- **Suggestion:** {suggestion}")
        md_parts.append("---")

    summary_md = f"""
### Overall Summary
- ✅ **Found & Standard:** {summary['Compliant']} items
- ⚠️ **Found but Deviates/Partial:** {summary['Partial']} items
- ❌ **Potentially Missing:** {summary['Missing']} items
- ❓ **Unclear/Not Found:** {summary['Unclear']} items
- 🔴 **High Risk Items/Errors:** {summary['High Risk']}
---
"""
    md_parts.insert(1, summary_md)
    return "\n".join(md_parts)

# --- Chatbot Q&A Function (Using RAG 2) --- RESTORED & CORRECTED ---
def answer_contextual_question_openai(
    user_question: str,
    general_qa_retriever: BaseRetriever, # Use RAG 2 retriever here
    ta_report: Optional[str] = None,
    past_messages: Optional[List[Dict[str, str]]] = None
) -> str:
    """ Answers user questions using RAG 2, the analysis report, and chat history. """
    global client # Use global synchronous client
    if not client or general_qa_retriever is None: # Check for RAG 2
        return "❌ Error: AI Resources (Client or RAG 2) not initialized."

    print(f"\n[Q&A] Answering with RAG 2: {user_question[:50]}...")

    # 1. Retrieve context from RAG 2 (General Q&A KB)
    try:
        retrieved_docs = general_qa_retriever.invoke(user_question)
        general_context_str = "\n---\n".join([doc.page_content for doc in retrieved_docs])
    except Exception as e:
        print(f"⚠️ Error retrieving RAG 2 context for Q&A: {e}")
        general_context_str = "Could not retrieve general Q&A context."

    # 2. Format Contexts
    report_context_str = ta_report if ta_report else "No specific document analysis report available for context."

    # 3. Define LLM Messages
    system_instruction = (
        "You are an expert assistant for Singapore Tenancy Agreements. "
        "Answer the user's FINAL QUESTION based *only* on the provided **General Q&A Context** (from RAG 2) and the **Analysis Report** (from RAG 1). "
        "Use the **Chat History** for conversation flow. If the question relates to a specific critique in the report, prioritize that. "
        "If the info isn't in context, say so clearly. Do not provide external legal advice."
    )
    messages = [{"role": "system", "content": system_instruction}]
    if past_messages: messages.extend(past_messages)

    # 4. Construct Final User Prompt
    final_user_prompt = textwrap.dedent(f"""
        CONTEXT FOR YOUR ANSWER:

        **1. ANALYSIS REPORT ON USER'S DOCUMENT (RAG 1 Results):**
        ---
        {report_context_str[:3000]} # Truncate report context if too long
        ---

        **2. GENERAL Q&A CONTEXT (RAG 2 Results):**
        ---
        {general_context_str[:3000]} # Truncate general context if too long
        ---

        **CHAT HISTORY:**
        ---
        {past_messages[-5:] if past_messages else "No history"}
        ---

        **FINAL USER QUESTION:** {user_question}

        Please provide the answer based *only* on the context above.
    """)
    messages.append({"role": "user", "content": final_user_prompt})

    # 5. Execute API Call
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL, messages=messages, temperature=0.0
            )
            answer_text = response.choices[0].message.content
            print("... Q&A answer generated.")
            return answer_text # Return raw text
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = 2 ** attempt
                print(f"OpenAI API Error (Q&A): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"OpenAI API failed (Q&A) after {MAX_RETRIES} attempts.")
                return f"❌ Unable to generate answer: {e}"
    return "❌ Unknown error during Q&A generation."

# --- Initialization Function for Streamlit --- UPDATED ---
def initialize_qa_resources(openai_api_key: Optional[str] = None, rebuild_rag1: bool = False, rebuild_rag2: bool = False) -> Tuple[Optional[BaseRetriever], Optional[BaseRetriever], Optional[Any]]:
    """
    Initializes/Loads RAG 1 & RAG 2 retrievers and the embeddings model.
    Checks for API key. Builds indices if needed.
    Returns (ideal_clauses_retriever, general_qa_retriever, embeddings_model)
    """
    global client, async_client, embeddings, ideal_clauses_retriever, general_qa_retriever

    # 1. Initialize OpenAI Clients if needed
    if not client and openai_api_key:
        print("Attempting to initialize OpenAI clients...")
        try:
            client = OpenAI(api_key=openai_api_key)
            async_client = AsyncOpenAI(api_key=openai_api_key)
            print("✅ OpenAI clients initialized.")
        except Exception as e:
            print(f"❌ Failed to initialize OpenAI clients: {e}")
            return None, None, None

    # 2. Initialize Embeddings if needed
    if embeddings is None:
        print("Attempting to initialize Embeddings model...")
        try:
            # Decide embedding model - using HuggingFace as per original
            embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
            print("✅ Embeddings model initialized.")
        except Exception as e:
             print(f"❌ Failed to initialize embeddings model: {e}")
             return None, None, None

    # 3. Build or Load RAG 1 Retriever
    if ideal_clauses_retriever is None:
        ideal_clauses_retriever = build_or_load_ideal_clauses_retriever(rebuild_index=rebuild_rag1)
        # Handle failure if needed

    # 4. Build or Load RAG 2 Retriever
    if general_qa_retriever is None:
        general_qa_retriever = build_or_load_general_qa_retriever(rebuild_index=rebuild_rag2)
        # Handle failure if needed

    # Return all initialized/loaded components
    # Ensure retrievers are not None before returning? Or handle downstream.
    if ideal_clauses_retriever is None or general_qa_retriever is None or embeddings is None:
         print("⚠️ Failed to initialize all required AI resources.")
         return None, None, None

    return ideal_clauses_retriever, general_qa_retriever, embeddings

# --- Helper function for User TA Loading/Chunking (Called from main.py) ---
def load_chunk_embed_user_ta(file_path: str, embeddings_model: object) -> List[Document]:
     """Loads, extracts, and chunks user TA PDF. Embeddings done later if needed."""
     # Note: Embeddings are done in pre-classification, no need to embed here.
     full_text = load_and_extract_pdf_text(file_path)
     user_chunks = split_user_document(full_text, source_name=os.path.basename(file_path))
     return user_chunks

# --- Placeholder/Removed Functions ---
# Removed: generate_ta_report_whole_doc and related functions.
# Removed: review_report (logging now integrated into run_checklist_analysis)
