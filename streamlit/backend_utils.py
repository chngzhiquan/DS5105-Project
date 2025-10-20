import pandas as pd
import numpy as np
import textwrap
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import json
import time
from typing import List, Dict, Optional, Any
from openai import OpenAI
from dotenv import load_dotenv
import os

# Get the directory of the currently executing file (backend_utils.py)
BACKEND_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- EMBEDDING MODEL CONFIGURATION ---
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
K = 3 # Number of top results to retrieve
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

# --- LLM CONFIGURATION for OpenAI ---
# Recommended model for this task (fast and good at structured output)
LLM_MODEL = "gpt-4o-mini" 
MAX_RETRIES = 3
load_dotenv()  # Load environment variables from a .env file if present
api_key_from_env = os.getenv("OPENAI_API_KEY") # Get the API key from the environment variable

if api_key_from_env:
    try:
        # Pass the key explicitly to ensure the client is initialized correctly
        client = OpenAI(api_key=api_key_from_env)
        print("OpenAI client initialized successfully from environment variable.")
    except Exception as e:
        print(f"FATAL ERROR: Failed to initialize OpenAI client even with key. Error: {e}")
        client = None
else:
    # If the key is not found, print a clear warning and set client to None
    print("OPENAI_API_KEY environment variable is NOT set.")
    client = None

# --- RAG 1: Ideal Clauses (The 'Gold Standard' for Comparison) ---
def build_ideal_clauses_retriever(data_directory="./TA_template", faiss_index_path="./faiss_index_ideal_clauses"):
    """
    Loads, chunks, and indexes the ideal tenancy agreement PDFs (RAG 1 source).
    Returns a LangChain FAISS Retriever.
    """
    print("\n--- BUILDING RAG 1: IDEAL CLAUSES RETRIEVER ---")
    
    # 1. Load Documents
    loader = DirectoryLoader(
        path=data_directory,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    all_documents = loader.load()
    print(f"Loaded {len(all_documents)} document pages.")

    # 2. Chunk Documents (using the custom, structure-aware splitter)
    custom_separators = [
        "\n\n",
        r"\n\s*[A-Z]+\s+\d*\s*\.",
        r"\n\s*\d+\.\d*\s*",
        r"\n\s*\([a-zA-Z0-9]+\)\s*",
        "\n", " ", ""
    ]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=custom_separators,
        is_separator_regex=True
    )
    all_chunks = text_splitter.split_documents(all_documents)
    print(f"Split into {len(all_chunks)} chunks.")

    print("\n--- IDEAL CLAUSES CHUNK PREVIEW (First 3) ---")
    for i, chunk in enumerate(all_chunks[:3]):
        print(f"Chunk {i+1} (Length: {len(chunk.page_content)}):")
        # Print the first 200 characters to keep the output manageable
        print(f"  {chunk.page_content[:200].replace('\n', ' ')}...") 
        print(f"  Metadata: {chunk.metadata}")
        print("-" * 20)


    # 3. Create Vector Store and Retriever
    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    print("FAISS index for Ideal Clauses created successfully.")

    # Save the index to the repository
    vectorstore.save_local(faiss_index_path)
    print(f"FAISS index saved to {faiss_index_path}.")

# --- RAG 2: General Q&A (Excel Source) ---
def build_general_qa_retriever(file_path, faiss_index_path="./faiss_index_general_qa"):
    """
    Loads data from the Excel file, converts it to Documents, and creates a FAISS retriever.
    """
    print("\n--- BUILDING RAG 2: GENERAL Q&A RETRIEVER ---")
    
    # 1. Load and process data (using the existing logic)
    try:
        df = pd.read_excel(file_path, header=1)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found. Using dummy data.")
        df = pd.DataFrame({
            'Question': ["How do I know if my tenancy agreement is valid?"],
            'Answer / Explanation': ["Must be in writing, signed by both parties, and include essential terms."],
            'Legal Commentary': ["Only agreements containing essential terms are enforceable."],
            'Government Regulation / Explanation': ["N/A"]
        })
    
    # 2. Convert Data Rows into LangChain Documents
    documents = []
    for index, row in df.iterrows():
        content = textwrap.dedent(f"""
            Question: {row.get('Question', 'N/A')}
            Answer/Explanation: {row.get('Answer / Explanation', 'N/A')}
            Legal Context: {row.get('Legal Commentary', 'N/A')}
            Regulation/Source: {row.get('Government Regulation / Explanation', 'N/A')}
        """).strip()
        
        # Create a LangChain Document
        doc = Document(
            page_content=content,
            metadata={"source": file_path, "row_index": index}
        )
        documents.append(doc)
        
    print(f"Created {len(documents)} General Q&A Documents.")

    print("\n--- GENERAL Q&A CHUNK PREVIEW (First 3) ---")
    for i, doc in enumerate(documents[:3]):
        print(f"Chunk {i+1}:")
        print(doc.page_content)
        print(f"  Metadata: {doc.metadata}")
        print("-" * 20)

    # 3. Create Vector Store and Retriever
    vectorstore = FAISS.from_documents(documents, embeddings)
    print("FAISS index for General Q&A created successfully.")
    
    # Save the index to the repository
    vectorstore.save_local(faiss_index_path)
    print(f"FAISS index saved to {faiss_index_path}.")

def load_retriever(FAISS_INDEX_PATH, K):
    """
    Loads the saved FAISS index and returns the retriever object for runtime use.
    """
    print("\n--- LOADING RAG RETRIEVER ---")
    
    if not os.path.exists(FAISS_INDEX_PATH):
        # This is a critical error if the index should be pre-built
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_INDEX_PATH}. "
            "Please run 'create_and_save_ideal_index()' first."
        )

    # 1. Load Vector Store
    # IMPORTANT: You still need the 'embeddings' model object to load the index
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True # Required by LangChain for loading
    )
    print("FAISS index loaded successfully.")

    # 2. Return Retriever
    return vectorstore.as_retriever(search_kwargs={"k": K})

def load_and_extract_pdf_text(file_path: str) -> str:
    """
    Checks if a file is a PDF, loads it, and extracts all text content.
    
    Args:
        file_path: The local path to the user's uploaded file.

    Returns:
        A single string containing all text from the PDF.

    Raises:
        ValueError: If the file is not found or is not a PDF.
    """
    print(f"\n--- Loading and Extracting Text from: {file_path} ---")

    # 1. Basic File Check
    if not os.path.exists(file_path):
        raise ValueError(f"Error: File not found at path: {file_path}")

    # 2. PDF Extension Check (Simple approach)
    if not file_path.lower().endswith('.pdf'):
        raise ValueError(f"Error: File is not a PDF ('.pdf' extension required).")

    try:
        # 3. Use LangChain's PyPDFLoader for robust text extraction
        loader = PyPDFLoader(file_path)
        
        # Load all pages as a list of Document objects
        pages = loader.load()
        print(f"Successfully loaded {len(pages)} pages.")

        # 4. Concatenate all page content into a single string
        full_text = "\n\n".join(page.page_content for page in pages)
        
        # Simple cleanup (optional, but helps with messy PDF parsing)
        full_text = re.sub(r'\s{2,}', ' ', full_text) # Replace multiple spaces/newlines with single space
        full_text = re.sub(r'(\n\s*){2,}', '\n\n', full_text) # Preserve paragraph breaks

        print("Text extraction complete.")
        return full_text
    
    except Exception as e:
        # Catch errors during PDF parsing
        raise RuntimeError(f"An error occurred during PDF text extraction: {e}")

def split_user_document(user_uploaded_text: str, source_name: str = "User TA") -> list[Document]:
    """
    Splits the raw text of the user-uploaded tenancy agreement into clause-level chunks.

    Args:
        user_uploaded_text: The raw string content of the user's document.
        source_name: A metadata tag to identify the source (e.g., the filename).

    Returns:
        A list of LangChain Document objects, one for each clause/chunk.
    """
    print(f"\n--- SPLITTING USER DOCUMENT: {source_name} ---")

    # The same structure-aware separators used for your Ideal Clauses (RAG 1)
    custom_separators = [
        "\n\n",
        r"\n\s*[A-Z]+\s+\d*\s*\.",
        r"\n\s*\d+\.\d*\s*",
        r"\n\s*\([a-zA-Z0-9]+\)\s*",
        "\n", " ", ""
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=custom_separators,
        is_separator_regex=True
    )

    # 1. Convert the single raw string into a list of Documents (one initial document)
    initial_document = [
        Document(page_content=user_uploaded_text, metadata={"source": source_name})
    ]

    # 2. Split the document based on the clause structure
    user_chunks = text_splitter.split_documents(initial_document)
    
    print(f"User TA split into {len(user_chunks)} clause-level chunks.")
    
    # Optional: Print a preview of the first few chunks
    for i, chunk in enumerate(user_chunks[:3]):
        print(f"  Chunk {i+1} (Length: {len(chunk.page_content)}): {chunk.page_content[:150].replace('\n', ' ')}...")
    
    return user_chunks

def llm_compare_and_critique_openai(
    user_clause_text: str, 
    ideal_context_docs: List[Document]
) -> Dict:
    """
    Compares a user's tenancy clause against retrieved ideal context using the OpenAI API.
    
    The function uses structured JSON output enforcement for reliability.

    Args:
        user_clause_text: The content of the user's clause chunk.
        ideal_context_docs: A list of relevant ideal clauses retrieved from the RAG 1 Vector Store.
        
    Returns:
        A dictionary containing the feedback, risk level, and suggestions.
    """
    if not client:
        return {
            "clause_summary": "API Initialization Failed",
            "risk_level": "HIGH",
            "feedback": "OpenAI client is not configured correctly. Check API key.",
            "suggestion": "Set the OPENAI_API_KEY environment variable."
        }

    print(f"\n[CRITIQUE] Analyzing clause with GPT: {user_clause_text[:50]}...")

    # 1. FORMAT THE CONTEXT FOR THE LLM
    context_str = "\n---\n".join([doc.page_content for doc in ideal_context_docs])
    
    # 2. DEFINE SYSTEM INSTRUCTION
    # This sets the persona and output constraints.
    system_instruction = (
        "You are a world-class legal analyst specializing in residential tenancy agreements. "
        "Your task is to compare a provided 'User Clause' against 'Ideal Clause Examples' "
        "and generate structured, actionable feedback. Be concise, professional, and focus only on deviations or missing protections for the tenant. "
        "Your entire output MUST be a single, valid JSON object that strictly adheres to the provided JSON Schema."
    )

    # 3. DEFINE THE USER QUERY (THE CORE PROMPT)
    user_query = textwrap.dedent(f"""
        Please analyze the following 'User Clause' and compare it to the 'Ideal Clause Examples'.

        **USER CLAUSE TO CRITIQUE:**
        ---
        {user_clause_text}
        ---

        **IDEAL CLAUSE EXAMPLES (RAG Context):**
        ---
        {context_str}
        ---

        Based on your comparison, provide the analysis in the specified JSON format.
    """)

    # 4. DEFINE THE JSON SCHEMA
    response_schema = {
        "type": "object",
        "properties": {
            "clause_summary": {"type": "string", "description": "A brief (1-sentence) summary of the user clause's main topic."},
            "risk_level": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"], "description": "The risk level (LOW, MEDIUM, or HIGH) for the tenant based on deviations from the ideal."},
            "feedback": {"type": "string", "description": "Specific, actionable criticism on what is missing or concerning in the User Clause."},
            "suggestion": {"type": "string", "description": "A brief sentence on how the user should attempt to modify the clause."}
        },
        "required": ["clause_summary", "risk_level", "feedback", "suggestion"]
    }


    # 5. EXECUTE API CALL WITH EXPONENTIAL BACKOFF
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_query}
                ],
                # Use the response_format tool for guaranteed JSON output (GPT-4o/GPT-4 models)
                response_format={"type": "json_object"}, 
                # Note: The JSON structure is also implicitly constrained by the prompt and system instruction.
                temperature=0.0 # Use low temperature for analytical tasks
            )
            
            # Extract and parse the JSON response text
            # The entire output text should be a single JSON string
            json_text = response.choices[0].message.content
            parsed_json = json.loads(json_text)
            print(f"... Successful critique generated on attempt {attempt + 1}.")
            return parsed_json
            
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = 2 ** attempt
                print(f"OpenAI API Error: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"OpenAI API failed after {MAX_RETRIES} attempts.")
                return {
                    "clause_summary": "Analysis Failed",
                    "risk_level": "HIGH",
                    "feedback": f"Unable to generate critique after {MAX_RETRIES} attempts. Error: {e}",
                    "suggestion": "Check your API key, model permissions, and rate limits."
                }
    
    # Should be unreachable
    return {}

def format_llm_report(data: List[Dict[str, Any]]):
    report_lines = ["# 📄 Comprehensive Tenancy Agreement Analysis Report\n"]
    issue_counter = 1
    
    for clause_feedback in data:
        # Check if the structure we expect (top-level 'feedback' key) is present
        feedback_data = clause_feedback.get('feedback', {})
        
        # We need a way to summarize *which clause* this is, but the raw output doesn't 
        # explicitly provide the clause text. We'll use a placeholder structure for now.
        report_lines.append(f"## 🛠️ Analysis of Clause {issue_counter}\n")
        
        has_content = False
        
        # 1. Process Deviations
        deviations = feedback_data.get('deviations', [])
        if deviations:
            report_lines.append("### 🔴 Deviations/High Risk Issues:\n")
            for dev in deviations:
                issue = dev.get('issue', 'N/A')
                description = dev.get('description', 'No description provided.')
                report_lines.append(f"* **{issue}:** {description}\n")
            has_content = True

        # 2. Process Missing Protections (or similar structures like vague_terms)
        missing_protections = feedback_data.get('missing_protections', [])
        if not missing_protections:
            # Check for other potential keys used by the LLM, like 'missingProtections' (camelCase)
            missing_protections = feedback_data.get('missingProtections', [])
        
        vague_terms = feedback_data.get('vague_terms', [])
        
        if missing_protections or vague_terms:
            report_lines.append("\n### 🟡 Missing Protections/Vague Terms:\n")
            
            # Combine all non-deviation issues for a clear list
            all_other_issues = missing_protections + vague_terms
            
            for issue_data in all_other_issues:
                # The keys change slightly here ('protection' or 'issue')
                issue = issue_data.get('protection') or issue_data.get('issue', 'N/A')
                description = issue_data.get('description', 'No description provided.')
                report_lines.append(f"* **{issue}:** {description}\n")
            has_content = True
        
        # 3. Add separator and increment counter
        if has_content:
            report_lines.append("\n---\n")
            issue_counter += 1
        
    # If the LLM output was perfectly fine, the list might be empty.
    if issue_counter == 1:
        return "# ✅ Analysis Complete: No significant deviations found in the provided clauses."
        
    return "".join(report_lines)

# --- RAG 1: Ideal Clauses Configuration ---
# Construct the absolute path: 
# Start at the script's dir, go up one (to DS5105-PROJECT), then find the target.
IDEAL_CLAUSES_FAISS = os.path.join(
    BACKEND_SCRIPT_DIR,
    "..",                     # Moves to DS5105-PROJECT/
    "faiss_index_ideal_clauses" # Finds the folder
)
K = 3

# --- GLOBAL RESOURCE INITIALIZATION (RAG 1) ---
try:
    # Assuming 'embeddings' is globally available and load_retriever is imported
    ideal_clauses_retriever = load_retriever(IDEAL_CLAUSES_FAISS, K)
except Exception as e:
    print(f"CRITICAL: Failed to load ideal clauses RAG index: {e}")
    ideal_clauses_retriever = None

# --- RAG 2: General Q&A Configuration ---
GENERAL_QA_FAISS = os.path.join(
    BACKEND_SCRIPT_DIR,
    "..",                     # Moves to DS5105-PROJECT/
    "faiss_index_general_qa" # Finds the folder
)
# K is already defined

# --- GLOBAL RESOURCE INITIALIZATION (RAG 2) ---
try:
    # Assuming 'embeddings' is globally available and load_retriever is imported
    general_qa_retriever = load_retriever(GENERAL_QA_FAISS, K)
except Exception as e:
    print(f"CRITICAL: Failed to initialize global RAG components: {e}")
    general_qa_retriever = None

# --------------------------------------------------------------------------------

# Function Integrating LLM and RAG 1 (Report Generation)
def generate_ta_report(USER_UPLOADED_FILE_PATH, ideal_clauses_retriever):
    """Generates the tenancy agreement analysis report using RAG 1 and LLM critique."""
    if ideal_clauses_retriever is None:
            raise RuntimeError("Report generation failed: Ideal RAG index not loaded.")
    try:
        # 1. LOAD & EXTRACT TEXT from the PDF
        full_user_document_text = load_and_extract_pdf_text(USER_UPLOADED_FILE_PATH)

        # 2. SPLIT the extracted text into clauses
        user_clause_chunks = split_user_document(
            full_user_document_text, 
            source_name=USER_UPLOADED_FILE_PATH
        )

        # 3. Loop through ALL user clauses for comparison (The RAG Core)
        feedback_report = []

        for user_clause in user_clause_chunks:
            # Use RAG 1 to retrieve the Ideal Clause context
            comparison_context = ideal_clauses_retriever.invoke(user_clause.page_content)
            
            # 4. LLM Call for Feedback (Simulated)
            feedback = llm_compare_and_critique_openai(user_clause.page_content, comparison_context)
            feedback_report.append(feedback)

        print("\n--- Phase 1 Complete: Analysis ready. ---")
        final_report = format_llm_report(feedback_report)
        return final_report # Changed this from final_report
        
    except (ValueError, RuntimeError) as e:
        # Handle the specific errors raised by the extraction function
        print(f"\nFATAL ERROR DURING FILE PROCESSING: {e}")
        # You would typically stop processing here and inform the user.

# Initialisation function for session state
def initialize_qa_resources(openai_api_key: str):
    """Initializes the OpenAI client (runtime dependency)."""
    
    # 1. Initialize OpenAI Client
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_api_key)
    except Exception as e:
        raise RuntimeError(f"OpenAI client configuration failed: {e}")

    # 2. Return the client and the globally initialized retriever
    if general_qa_retriever is None:
        raise RuntimeError("General QA Retriever failed to load during module import.")
        
    return client, general_qa_retriever

# Integration of LLM and RAG 2
def answer_contextual_question_openai(
    user_question: str, 
    general_qa_retriever: object,
    ta_report: Optional[List[Dict]] = None, # Optional TA Report (Phase 1 output)
    past_messages: Optional[List[Dict[str, str]]] = None # Optional Chat History
) -> str:
    """
    Answers a user question using RAG 2, the TA report, and chat history.
    """
    if not client:
        return "OpenAI client is not configured correctly. Check API key."

    print(f"\n[Q&A] Answering contextual question with GPT: {user_question[:50]}...")

    # 1. Use RAG 2 to retrieve the General Q&A context
    qa_context = general_qa_retriever.invoke(user_question)
    
    # 2. Format Contexts
    general_context_str = "\n---\n".join([doc.page_content for doc in qa_context])
    report_context_str = str(ta_report) if ta_report else "No specific document analysis report available." # Format report if provided

    # 3. DEFINE THE LLM MESSAGE LIST (Chat History Integration)
    
    # Update System Instruction to acknowledge all contexts
    system_instruction = (
        "You are an expert legal assistant specializing in residential tenancy agreements. "
        "Your response must be based on the provided context, which includes the **General Law Context** (RAG data) "
        "and, if present, the **User Document Analysis Report** (specific critiques). "
        "Maintain the flow of the conversation history. If the user's question relates to a flagged clause, "
        "prioritize the information from the Analysis Report. Be concise and professional."
    )
    
    messages = [
        {"role": "system", "content": system_instruction}
    ]

    # Add past conversation messages if they exist
    if past_messages:
        messages.extend(past_messages) # Extends the list with previous turns

    # 4. Construct the Final User Query (containing all RAG context)
    final_user_prompt = textwrap.dedent(f"""
        Please answer the FINAL USER QUESTION using the context provided below.

        **THIS IS A REPORT ON THE USER'S TENANCY AGREEMENT (Specific Critiques):**
        ---
        {report_context_str}
        
        **GENERAL Q&A CONTEXT (RAG Retrieved):**
        ---
        {general_context_str}
        ---

        **FINAL USER QUESTION:** {user_question}
    """)
    
    # Add the final, context-rich user prompt
    messages.append({"role": "user", "content": final_user_prompt})

    # 5. EXECUTE API CALL WITH EXPONENTIAL BACKOFF
    for attempt in range(MAX_RETRIES):
        try:
         
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages, # Now uses the full history list
                temperature=0.0
            )
            answer_text = response.choices[0].message.content
            styled_answer = f"""<span style="color:black;">{answer_text}</span>""" 
            
            print(f"... Successful answer generated on attempt {attempt + 1}.")
            return styled_answer
            
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = 2 ** attempt
                print(f"OpenAI API Error: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"OpenAI API failed after {MAX_RETRIES} attempts.")
                return f"Unable to generate answer after {MAX_RETRIES} attempts. Error: {e}"
    
    return "Unknown error."