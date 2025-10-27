#!/usr/bin/env python3
"""
OneCheck&Chat - Tenancy Agreement Analyzer (v3 - Checklist RAG with Pre-classification)
AI-powered tenancy agreement verification and chat system using Checklist-Guided RAG.

Run with: streamlit run main.py
"""

import streamlit as st
import os
from typing import Optional, List, Dict, Any
import tempfile
import base64
from datetime import datetime

# --- Import Backend Functions ---
# Ensure these functions exist in your backend_utils.py (with latest updates)
try:
    from backend_utils import (
        checklist,
        preclassify_chunks_by_similarity,
        load_chunk_embed_user_ta,
        run_checklist_analysis, # Performs checklist RAG loop + validation logging
        format_analysis_results_to_markdown,
        answer_contextual_question_openai, # Uses RAG 2
        initialize_qa_resources # Initializes RAG 1, RAG 2, embeddings
    )
    BACKEND_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import backend functions: {e}. Ensure backend_utils.py is correct.")
    BACKEND_AVAILABLE = False
    # Define stubs
    checklist = []
    def preclassify_chunks_by_similarity(*args, **kwargs): return {}
    def load_chunk_embed_user_ta(*args, **kwargs): return []
    def run_checklist_analysis(*args, **kwargs): return []
    def format_analysis_results_to_markdown(*args, **kwargs): return "Backend not available."
    def answer_contextual_question_openai(*args, **kwargs): return "Backend not available."
    def initialize_qa_resources(*args, **kwargs): return None, None, None # retriever1, retriever2, embeddings

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# Core libraries (check if needed directly)
try:
    from langchain_community.document_loaders import PyPDFLoader
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# --- Page Configuration ---
st.set_page_config(
    page_title="OneCheck&Chat",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Styling ---
def setup_custom_css():
    """Configure custom CSS for better styling"""
    # (Keep CSS styles as they were in your original script)
    st.markdown("""
    <style>
    .main-header { font-size: 3rem; color: #2E86AB; text-align: center; margin-bottom: 1rem; font-weight: bold; }
    .subtitle { font-size: 1.2rem; color: #666; text-align: center; margin-bottom: 2rem; }
    .section-header { background: linear-gradient(90deg, #2E86AB 0%, #A23B72 100%); color: white; padding: 1rem; border-radius: 10px; margin: 1rem 0; font-size: 1.3rem; font-weight: bold; }
    .result-box { background-color: #f8f9fa; border-left: 5px solid #2E86AB; padding: 1.5rem; border-radius: 8px; margin: 1rem 0; color: black; }
    .chat-message { padding: 1rem; border-radius: 10px; margin: 0.5rem 0; color: black; }
    .user-message { background-color: #E3F2FD; border-left: 4px solid #2196F3; }
    .assistant-message { background-color: #F3E5F5; border-left: 4px solid #9C27B0; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: 500; }
    .upload-section { background-color: #f0f8ff; padding: 2rem; border-radius: 12px; border: 2px dashed #2E86AB; color: #000000; }
    </style>
    """, unsafe_allow_html=True)

# --- Session State Initialization ---
def initialize_session_state():
    """Initialize session state variables"""
    # File upload state
    if "uploaded_file_name" not in st.session_state: st.session_state.uploaded_file_name = None
    if "uploaded_file_content" not in st.session_state: st.session_state.uploaded_file_content = None
    # Processed user TA state
    if "user_chunks" not in st.session_state: st.session_state.user_chunks = None
    if "classification_map" not in st.session_state: st.session_state.classification_map = None
    # RAG resources state
    if "ideal_clauses_retriever" not in st.session_state: st.session_state.ideal_clauses_retriever = None # RAG 1
    if "general_qa_retriever" not in st.session_state: st.session_state.general_qa_retriever = None # RAG 2 (Restored)
    if "embeddings_model" not in st.session_state: st.session_state.embeddings_model = None
    # Analysis results state
    if "verification_results" not in st.session_state: st.session_state.verification_results = None
    # Chat state
    if "messages" not in st.session_state: st.session_state.messages = []
    # UI state
    if "is_processing" not in st.session_state: st.session_state.is_processing = False

# --- Sidebar ---
def create_sidebar():
    """Create sidebar with configuration and info"""
    with st.sidebar:
        st.image("https://via.placeholder.com/200x80/2E86AB/FFFFFF?text=OneCheck%26Chat", use_container_width=True)
        st.markdown("---")
        st.subheader("🔑 API Configuration")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            api_key = st.text_input("OpenAI API Key:", type="password", help="Required for AI analysis")
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
            else:
                st.warning("⚠️ Please enter your OpenAI API Key")
                st.stop()

        # Initialize backend resources if key is available and resources aren't loaded
        resources_loaded = (st.session_state.ideal_clauses_retriever is not None and
                            st.session_state.general_qa_retriever is not None and
                            st.session_state.embeddings_model is not None)

        if api_key and not resources_loaded:
             with st.spinner("Initializing AI Resources (RAG 1, RAG 2, Embeddings)..."):
                try:
                    # initialize_qa_resources should return all three now
                    retriever1, retriever2, embeddings_model = initialize_qa_resources(api_key)
                    if retriever1 and retriever2 and embeddings_model:
                        st.session_state.ideal_clauses_retriever = retriever1
                        st.session_state.general_qa_retriever = retriever2 # Store RAG 2
                        st.session_state.embeddings_model = embeddings_model
                        st.success("✅ AI Resources Initialized")
                    else:
                        st.error("❌ Failed to initialize one or more AI resources.")
                        st.stop()
                except Exception as e:
                    st.error(f"⚠️ AI Initialization failed: {e}")
                    st.stop()
        elif resources_loaded:
             st.success("✅ AI Resources Loaded")
        elif not api_key:
             st.warning("Enter API Key to initialize AI resources.")


        # --- REMOVED Analysis Engine Selection ---

        st.markdown("---")
        if st.session_state.uploaded_file_name:
            st.subheader("📄 Current Document")
            st.info(f"**{st.session_state.uploaded_file_name}**")
            if st.button("🗑️ Clear Document & Reset", type="secondary"):
                keys_to_reset = [
                    "uploaded_file_name", "uploaded_file_content", "user_chunks",
                    "classification_map", "verification_results", "messages"
                ]
                for key in keys_to_reset:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

        st.markdown("---")
        with st.expander("ℹ️ How to Use"):
             st.markdown("""
             1. **Upload** tenancy agreement PDF.
             2. **Process** document (loads, chunks, classifies).
             3. **Analyze** contract against checklist (RAG).
             4. **Chat** with AI about the agreement or general questions.
             """)
        st.markdown("---")
        st.caption("**OneCheck&Chat v3.0**") # Version bump
        st.caption("© 2025 Capstone Team")

# --- Document Processing ---
def process_uploaded_document(uploaded_file) -> bool:
    """Process the uploaded PDF: load, chunk, and pre-classify"""
    try:
        st.session_state.uploaded_file_content = uploaded_file.getvalue()
        file_extension = os.path.splitext(uploaded_file.name)[1]
        temp_file_path = None

        with st.spinner("Processing document: Loading & Chunking..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                tmp_file.write(st.session_state.uploaded_file_content)
                temp_file_path = tmp_file.name

            embeddings_model = st.session_state.get("embeddings_model")
            if not embeddings_model:
                 st.error("Embeddings model not initialized!")
                 return False

            # Load and chunk the document using backend function
            user_chunks = load_chunk_embed_user_ta(temp_file_path, embeddings_model)
            st.session_state.user_chunks = user_chunks

        if st.session_state.user_chunks:
             with st.spinner("Pre-classifying document sections using embeddings..."):
                 classification_map = preclassify_chunks_by_similarity(
                     user_chunks=st.session_state.user_chunks,
                     checklist=checklist,
                     embeddings_model=embeddings_model,
                     similarity_threshold=0.70 # Tune this
                 )
                 st.session_state.classification_map = classification_map
                 return True
        else:
             st.error("❌ Failed to chunk document.")
             return False

    except Exception as e:
        st.error(f"Error during document processing: {str(e)}")
        st.session_state.user_chunks = None
        st.session_state.classification_map = None
        return False
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

# --- UI Sections ---
def create_upload_section():
    st.markdown('<div class="section-header">📤 1. Upload Tenancy Agreement</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a PDF file:", type="pdf", key="pdf_uploader"
    )

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        with col1: st.metric("📄 File Name", uploaded_file.name)
        with col2: st.metric("📊 Size", f"{uploaded_file.size / 1024:.1f} KB")

        # Show Process button only if file is new or not yet processed
        if (st.session_state.uploaded_file_name != uploaded_file.name or
            st.session_state.classification_map is None): # Check classification map status
            if st.button("🔄 Process Document & Prepare for Analysis", type="primary"):
                success = process_uploaded_document(uploaded_file)
                if success:
                    st.session_state.uploaded_file_name = uploaded_file.name
                    st.success(f"✅ Document processed & pre-classified ({len(st.session_state.user_chunks)} chunks). Ready for analysis.")
                    st.rerun()
        else:
             st.success(f"✅ Document '{uploaded_file.name}' is ready.")
    else:
        st.info("👆 Please upload a PDF file.")

def create_contract_verification_section():
    st.markdown('<div class="section-header">🤖 2. AI Contract Analysis (Checklist-Guided RAG)</div>', unsafe_allow_html=True)
    if not st.session_state.get("classification_map"): # Check if pre-classification done
        st.info("📋 Please upload and process a document first.")
        return

    st.markdown("""
    <div class="result-box">
    <h4>⚖️ Tenancy Agreement Compliance Check</h4>
    <p>Click 'Analyze Contract' to compare your document against standard clauses using our AI checklist. Results include validation scores logged in the backend.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("▶️ Analyze Contract", type="primary"):
        # Check prerequisites
        if not st.session_state.classification_map or not st.session_state.ideal_clauses_retriever:
            st.error("❌ Prerequisites not met. Ensure document processed & AI resources loaded.")
            return

        with st.spinner("Analyzing contract using Checklist-Guided RAG... (May take ~1 min)"):
            try:
                # Call the backend function for checklist-guided RAG + validation logging
                analysis_report_list = run_checklist_analysis(
                    classification_map=st.session_state.classification_map,
                    ideal_clauses_retriever=st.session_state.ideal_clauses_retriever,
                    checklist=checklist
                )
                # Format results for display
                formatted_report = format_analysis_results_to_markdown(analysis_report_list)
                st.session_state.verification_results = formatted_report
                st.success("✅ Contract analysis complete! Validation data logged.")
                # Don't rerun, let results display below
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.session_state.verification_results = None

    # Display results
    if st.session_state.verification_results:
        st.markdown("---")
        st.markdown("**Analysis Results:**")
        st.markdown(st.session_state.verification_results)

def create_chat_section():
    st.markdown('<div class="section-header">💬 3. Chat About Your Agreement / General Q&A</div>', unsafe_allow_html=True)
    if not st.session_state.get("uploaded_file_name"):
        st.info("📋 Upload and ideally analyze a document first for best chat context.")
        # Allow chat even without analysis, but context will be limited
        # return # Optionally block chat until analysis is run

    st.subheader("💭 Ask Specific Questions")

    chat_container = st.container(height=400)
    with chat_container:
        if st.session_state.messages:
            for message in st.session_state.messages:
                role_class = "user-message" if message["role"] == "user" else "assistant-message"
                icon = "👤" if message["role"] == "user" else "🤖"
                # Use st.markdown with unsafe_allow_html=True to render potential markdown in responses
                st.markdown(f'<div class="chat-message {role_class}"><strong>{icon} {message["role"].capitalize()}:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
        else:
             st.caption("Ask a question about your TA analysis or general tenancy topics...")

    user_question = st.chat_input("Ask about a clause, term, or general question...")
    if user_question:
        handle_user_question(user_question)

def handle_user_question(question: str):
    """Handle user question input for chat using RAG 2"""
    # Use RAG 2 retriever for general Q&A
    general_retriever = st.session_state.get("general_qa_retriever")
    if not general_retriever:
        st.error("❌ RAG 2 (General Q&A) Resources not initialized.")
        return

    st.session_state.messages.append({"role": "user", "content": question})

    chat_history_for_llm = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages[:-1]
        if msg["role"] in ("user", "assistant")
    ]

    with st.spinner("Thinking..."):
        try:
            # Call backend Q&A function - Ensure it uses general_qa_retriever
            response = answer_contextual_question_openai(
                user_question=question,
                general_qa_retriever=general_retriever, # Pass RAG 2
                ta_report=st.session_state.verification_results, # Pass RAG 1 analysis report
                past_messages=chat_history_for_llm
            )
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"❌ Chatbot response failed: {str(e)}")
            st.session_state.messages.pop()
    st.rerun()

# --- Main Application Logic ---
def main():
    """Main application function"""
    setup_custom_css()
    initialize_session_state()

    st.markdown('<h1 class="main-header">📋 OneCheck&Chat</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI Tenancy Agreement Analyzer (Checklist RAG)</p>', unsafe_allow_html=True)

    create_sidebar() # Creates sidebar and initializes AI resources

    if not BACKEND_AVAILABLE or not LANGCHAIN_AVAILABLE:
        st.error("⚠️ Required libraries missing or backend import failed.")
        st.stop()

    # Ensure AI resources are loaded before proceeding
    resources_ready = (st.session_state.get("ideal_clauses_retriever") and
                       st.session_state.get("general_qa_retriever") and
                       st.session_state.get("embeddings_model"))
    if not resources_ready:
         st.warning("⏳ Waiting for AI Resources initialization in the sidebar...")
         st.stop()


    st.markdown("---")
    create_upload_section()
    st.markdown("---")
    create_contract_verification_section()
    st.markdown("---")
    create_chat_section()
    st.markdown("---")
    # Placeholder sections (can be uncommented when implemented)
    # create_rag_verification_section()
    # st.markdown("---")
    # create_export_section()

    st.markdown("---")
    st.markdown('<div style="text-align: center; color: #666; padding: 2rem;"><p><strong>OneCheck&Chat v3.0</strong></p></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    if BACKEND_AVAILABLE:
        main()
    else:
        st.error("Application cannot start due to backend import errors.")

```

**Key Changes in `main.py`:**

  * **Imports:** Adjusted to reflect the new/renamed backend functions (`run_checklist_analysis`, etc.) and the restored RAG 2 logic.
  * **Initialization:** `initialize_session_state` and `create_sidebar` ensure both `ideal_clauses_retriever` (RAG 1) and `general_qa_retriever` (RAG 2) are loaded into the session state.
  * **Document Processing:** `process_uploaded_document` now includes the call to `preclassify_chunks_by_similarity` and stores the `classification_map`.
  * **Contract Analysis:** The "Analyze Contract" button logic now calls `run_checklist_analysis`, passing the `classification_map` and RAG 1 retriever. The validation logging happens automatically within this backend function.
  * **Chat Section:** `handle_user_question` correctly uses `general_qa_retriever` (RAG 2) again, passing it to `answer_contextual_question_openai`.
  * **Removed Old Logic:** All references to `generate_ta_report_whole_doc`, `analysis_mode`, `checklist_path`, etc., have been removed.

This version should now correctly implement the checklist-guided RAG with pre-classification and validation logging for the main analysis, while keeping the separate RAG 2 system for general Q\&A during the chat phase.
