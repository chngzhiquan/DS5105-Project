#!/usr/bin/env python3

import streamlit as st
import os
from typing import Optional, List, Dict, Any
import tempfile
import base64
from datetime import datetime
from backend_utils_v2 import (
    generate_ta_report,
    answer_contextual_question_openai,
    ideal_clauses_retriever,
    general_qa_retriever,
    review_report, 
    initialize_qa_resources,
    generate_ta_report_whole_doc,
    translate_document
)

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# Core libraries for PDF processing and AI
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.chat_models import ChatOpenAI
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory
    import openai
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="LeaseOwl",
    page_icon=" 🦉",
    layout="wide",
    initial_sidebar_state="expanded"
)

def setup_custom_css():
    """Configure custom CSS for better styling"""
    st.markdown("""
    <style>
    /* Header */
    .main-header {
        font-size: 3rem;
        color: #0D3B66; /* dark blue */
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #1E6091; /* medium blue */
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Section header with blue gradient */
    .section-header {
        background: linear-gradient(90deg, #1E6091 0%, #4DA8DA 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-size: 1.3rem;
        font-weight: bold;
    }

    /* Result boxes */
    .result-box {
        background-color: #D0E6F7; /* light blue */
        border-left: 5px solid #1E6091; /* medium blue */
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #000;
    }
    .warning-box {
        background-color: #FFF4E6;
        border-left: 5px solid #FFB400; /* yellow accent */
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #D6F5D6; /* soft green */
        border-left: 5px solid #0D3B66;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    /* Chat messages */
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #A9D6E5; /* light blue */
        border-left: 4px solid #0D3B66; /* dark blue */
        color: #000000;
    }
    .assistant-message {
        background-color: #4DA8DA; /* medium blue */
        border-left: 4px solid #0D3B66; /* dark blue */
        color: #FFFFFF;
    }

    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 500;
        background-color: #1E6091;
        color: white;
    }
    .stButton>button:hover {
        background-color: #0D3B66;
        color: white;
    }

    /* Upload section */
    .upload-section {
        background-color: #E0F1FA; /* very light blue */
        padding: 2rem;
        border-radius: 12px;
        border: 2px dashed #1E6091;
        color: #000000;
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    
    # File upload state
    if "uploaded_file_name" not in st.session_state:
        st.session_state.uploaded_file_name = None
    
    if "uploaded_file_content" not in st.session_state:
        st.session_state.uploaded_file_content = None
    
    # RAG state
    # Ideal Clauses RAG retriever object (RAG 1)
    if "ideal_clauses_retriever" not in st.session_state:
        st.session_state.ideal_clauses_retriever = ideal_clauses_retriever

    # General Q&A RAG retriever object (RAG 2)
    if "general_qa_retriever" not in st.session_state:
        st.session_state.general_qa_retriever = general_qa_retriever
    
    if "rag_results" not in st.session_state:
        st.session_state.rag_results = None
    
    # Contract verification state
    if "verification_results" not in st.session_state:
        st.session_state.verification_results = None
    
    # Chat state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Export state
    if "export_preview" not in st.session_state:
        st.session_state.export_preview = None
    
    # UI state
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    
    if "current_tab" not in st.session_state:
        st.session_state.current_tab = "upload"

def create_sidebar():
    """Create sidebar with configuration and info"""
    
    with st.sidebar:
        st.image("Logo.jpeg", use_container_width=True)
        
        st.markdown("---")
        
        # API Key management
        st.subheader("🔑 API Configuration")
        
        env_api_key = os.getenv("OPENAI_API_KEY")
        
        if env_api_key:
            st.success("✅ API Key loaded")
            # Initialize OpenAI client for backend utils
            try:
                initialize_qa_resources(env_api_key)
            except Exception as _e:
                st.warning(f"⚠️ OpenAI init failed: {_e}")
        else:
            api_key = st.text_input(
                "OpenAI API Key:",
                type="password",
                help="Required for AI analysis"
            )
            
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                # Initialize OpenAI client for backend utils
                try:
                    initialize_qa_resources(api_key)
                except Exception as _e:
                    st.warning(f"⚠️ OpenAI init failed: {_e}")
                st.success("✅ API Key configured")
            else:
                st.warning("⚠️ Please enter your OpenAI API Key")
        
        st.markdown("---")

        # === Analysis Engine ===
        st.subheader("Analysis Engine")
        analysis_mode = st.radio(
            "Choose engine",
            ["Fast", "Indexed (RAG)"],
            index=0,
            key="analysis_mode_radio"
        )
        st.session_state.analysis_mode = analysis_mode  

        # Checklist path (only for Whole-Doc)
        default_checklist = os.path.expanduser("./checklist/checklist.csv") 
        checklist_path = st.text_input(
            "Checklist file (.csv)",
            value=default_checklist,
            help="Used only in Fast (Whole-Doc) mode",
            key="checklist_path_input"
        )
        st.session_state.checklist_path = checklist_path

        st.markdown("---")
        
        # Document info
        if st.session_state.uploaded_file_name:
            st.subheader("📄 Current Document")
            st.info(f"**{st.session_state.uploaded_file_name}**")
            
            if st.button("🗑️ Clear Document", type="secondary"):
                # Reset all states
                st.session_state.uploaded_file_name = None
                st.session_state.uploaded_file_content = None
                st.session_state.rag_results = None
                st.session_state.verification_results = None
                st.session_state.messages = []
                st.session_state.export_preview = None
                st.rerun()
        
        st.markdown("---")
        
        # Help section
        with st.expander("ℹ️ How to Use"):
            st.markdown("""
            ### Steps:
            1. **Upload** tenancy agreement PDF
            2. **Translate** to your language of choice
            3. **Check** AI contract analysis
            4. **Chat** with the Chatbot
            5. **Export** results
            
            ### Requirements:
            - OpenAI API key
            - PDF tenancy agreement
            """)
        
        st.markdown("---")
        
        # About
        st.caption("**LeaseOwl v1.0**")
        st.caption("Tenancy Agreement Analyzer")
        st.caption("© 2025 All rights reserved")

def create_upload_section():
    """Section 1: Upload PDF file"""
    
    st.markdown('<div class="section-header">📤 1. Upload Tenancy Agreement</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file of your tenancy agreement:",
        type="pdf",
        help="Upload the tenancy agreement document for analysis",
        key="pdf_uploader"
    )
    
    if uploaded_file is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📄 File Name", uploaded_file.name)
        with col2:
            st.metric("📊 Size", f"{uploaded_file.size / 1024:.1f} KB")
        with col3:
            st.metric("📅 Uploaded", datetime.now().strftime("%H:%M:%S"))
        
        if st.session_state.uploaded_file_name != uploaded_file.name:
            if st.button("🔄 Process Document", type="primary"):
                with st.spinner("🔄 Processing document..."):
                    success = process_uploaded_document(uploaded_file)
                    if success:
                        st.session_state.uploaded_file_name = uploaded_file.name
                        st.success("✅ Document processed successfully!")
                        st.balloons()
                    else:
                        st.error("❌ Failed to process document")
        else:
            st.success(f"✅ Document ready: **{uploaded_file.name}**")
    else:
        st.info("👆 Please upload a PDF file to begin analysis")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return uploaded_file

def process_uploaded_document(uploaded_file) -> bool:
    """Process the uploaded PDF document"""
    
    try:
        # Save file content
        st.session_state.uploaded_file_content = uploaded_file.getvalue()
        
        return True
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return False

def create_translation_section():
    """Section 2: Document Translation"""
    
    st.markdown('<div class="section-header">🌐 2. Translation</div>', unsafe_allow_html=True)
    
    if not st.session_state.uploaded_file_name:
        st.info("📋 Upload a document first to enable translation")
        return
    
    st.markdown("""
    <div class="result-box" style="color:black;">
    <h4>🈸 Translate Document</h4>
    <p>Translate your tenancy agreement into a target language using AI.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Language selection
    target_language = st.selectbox(
        "Select target language:",
        ["English", "Indonesian", "Chinese", "Spanish", "French", "German"],
        index=1,
        key="target_language"  # store in session_state automatically
    )
    
    # Translate button inside function
    if st.button("🌍 Translate Document", type="primary"):
        uploaded_content = st.session_state.get('uploaded_file_content')
        if not uploaded_content:
            st.error("❌ No document content available for translation")
        else:
            with st.spinner(f"Translating document to {st.session_state.target_language}..."):
                try:
                    # LANGUAGE MAPPING
                    language_map = {
                        "Chinese": "Mandarin",
                        "English": "English",
                        "Indonesian": "Indonesian",
                        "Spanish": "Spanish",
                        "French": "French",
                        "German": "German"
                    }
                    mapped_language = language_map.get(st.session_state.target_language, st.session_state.target_language)

                    # Save temp file
                    suffix = ".pdf" if st.session_state.uploaded_file_name.lower().endswith(".pdf") else ".docx"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                        tmp_file.write(uploaded_content)
                        tmp_file_path = tmp_file.name

                    translations = translate_document(tmp_file_path, [mapped_language])
                    
                    if "error" in translations:
                        st.error(f"❌ Translation failed: {translations['error']}")
                        st.session_state.translated_text = None 
                    else:
                        st.session_state.translated_text = translations.get(mapped_language, "Translation failed (Key not found after success)")
                        st.success(f"✅ Document translated to {st.session_state.target_language}!")

                except Exception as e:
                    st.error(f"❌ Translation failed: {str(e)}")
                    st.session_state.translated_text = None 

    # Display translation + download
    translated_text = st.session_state.get("translated_text")
    if translated_text:
        st.markdown("---")
        st.markdown(f"**📄 Translation Preview ({st.session_state.target_language}):**")
        st.text_area("Translated Document", translated_text, height=400)

        st.download_button(
            label=f"📥 Download Translated Document ({st.session_state.target_language})",
            data=translated_text,
            file_name=f"{st.session_state.uploaded_file_name}_translated_{st.session_state.target_language}.txt",
            mime="text/plain"
        )

def create_contract_verification_section():
    """Section 3: AI Contract Verification"""
    
    st.markdown('<div class="section-header">🤖 3. AI Contract Analysis</div>', unsafe_allow_html=True)
    
    if not st.session_state.uploaded_file_name:
        st.info("📋 Upload a document first to enable contract analysis")
        return
    
    st.markdown("""
    <div class="result-box" style="color:black;">
    <h4>⚖️ Tenancy Agreement Compliance Check</h4>
    <p>AI-powered analysis of contract terms, conditions, and legal compliance.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Checklist of what to verify
        st.write("**Analysis Checklist:**")
        
        checks = [
            "Lease duration and renewal terms",
            "Rent amount and payment schedule",
            "Security deposit requirements",
            "Tenant and landlord responsibilities",
            "Maintenance and repair obligations",
            "Termination and notice periods",
            "Prohibited activities",
            "Legal compliance with local laws"
        ]
        
        for check in checks:
            st.checkbox(check, key=f"check_{check}", disabled=True)
        
        # Code space for AI verification
        with st.expander("🔧 AI Verification Code Space", expanded=False):
            st.code("""
# === CONTRACT VERIFICATION CODE ===
# TODO: Implement AI contract verification logic here

def analyze_contract_terms(document_text, llm):
    '''
    Analyze tenancy agreement for key terms and compliance
    
    Args:
        document_text: The full text of the contract
        llm: Language model for analysis
        
    Returns:
        dict: Analysis results with findings
    '''
    
    analysis_prompt = '''
    Analyze this tenancy agreement and provide:
    1. Key terms (rent, duration, deposit)
    2. Tenant obligations
    3. Landlord obligations
    4. Potential issues or unfair terms
    5. Missing standard clauses
    6. Legal compliance concerns
    '''
    
    # TODO: Implement actual AI analysis
    
    results = {
        "key_terms": {},
        "obligations": {},
        "issues": [],
        "compliance": "pending"
    }
    
    return results

# Run analysis
if st.session_state.conversation_chain:
    results = analyze_contract_terms(document_text, llm)
    st.session_state.verification_results = results
            """, language="python")
    
    with col2:
        if st.button("▶️ Analyze Contract", type="primary"):
            uploaded_content = st.session_state.get('uploaded_file_content')
            if not uploaded_content:
                st.error("❌ No document content available for analysis")
                return
            # Initialise temporary file
            temp_file_path = None
            with st.spinner("Analyzing contract..."):
                try:
                    file_extension = os.path.splitext(st.session_state.get('uploaded_file_name','.pdf'))[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                        tmp_file.write(uploaded_content)
                        temp_file_path = tmp_file.name
                        
                    analysis_mode = st.session_state.get("analysis_mode", "Fast (Whole-Doc, No Index)")
                    if str(analysis_mode).startswith("Fast"):
                        # Whole-Doc (no RAG)
                        checklist_path = st.session_state.get("checklist_path", "./checklist/checklist.csv")

                        if not os.path.exists(checklist_path):
                            st.error(f"Checklist not found: {checklist_path}")
                            return

                        report_md, raw_json, translations = generate_ta_report_whole_doc(
                            USER_UPLOADED_FILE_PATH=temp_file_path,
                            checklist_path=checklist_path,
                        )
                        st.session_state.verification_results = report_md
                        st.session_state.rag_results = None
                        st.success("✅ Contract analysis completed! (Whole-Doc)")
                    else:
                        # Indexed RAG1 (original flow)
                        from backend_utils import generate_ta_report, review_report
                        ideal_retriever = st.session_state.get("ideal_clauses_retriever", None)
                        if ideal_retriever is None:
                            st.error("RAG1 index not loaded. Please build/load the Ideal Clauses index.")
                            return

                        report, review_prompt = generate_ta_report(
                            USER_UPLOADED_FILE_PATH=temp_file_path,
                            ideal_clauses_retriever=ideal_retriever
                        )
                        st.session_state.verification_results = report
                        st.session_state.rag_results = review_report(review_prompt)
                        st.success("✅ Contract analysis completed! (RAG1)")
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.session_state.verification_results = None
                    st.session_state.rag_results = None
                finally:
                    if temp_file_path and os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
    
    # Display results if available
    if st.session_state.verification_results:
        st.markdown("---")
        st.markdown("**Analysis Results:**")
        
        # Placeholder for results display
        result_tabs = st.tabs(["📋 Summary", "⚠️ Issues", "✅ Compliance", "📊 Details"])
        
        with result_tabs[0]:
            st.markdown(st.session_state.verification_results)

        with result_tabs[1]:
            st.markdown(st.session_state.rag_results)
        
        with result_tabs[2]:
            st.info("Compliance status will be displayed here")
        
        with result_tabs[3]:
            st.info("Detailed analysis will be displayed here")

def create_chat_section():
    """Section 4: Chatbot"""
    
    st.markdown('<div class="section-header">💬 4. Chatbot</div>', unsafe_allow_html=True)
    
    if not st.session_state.uploaded_file_name:
        st.info("📋 Upload a document first to enable chat")
        return
    
    # Code space for chatbot
    with st.expander("🔧 Chatbot Code Space", expanded=False):
        st.code("""
# === CHATBOT CODE ===
# TODO: Implement chatbot logic here

def setup_chatbot(vectorstore):
    '''
    Setup conversational AI for document Q&A
    
    Args:
        vectorstore: Vector database with document embeddings
        
    Returns:
        ConversationalRetrievalChain: Chat chain
    '''
    
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.7
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    
    return chain

# Initialize chatbot
if st.session_state.vectorstore and not st.session_state.conversation_chain:
    st.session_state.conversation_chain = setup_chatbot(st.session_state.vectorstore)
        """, language="python")
    
    st.markdown("---")
    
    # Chat interface
    st.subheader("💭 Ask Questions About Your Agreement")
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        if st.session_state.messages:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message" style="color:black;">
                        <strong>👤 You:</strong><br>
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message assistant-message" style="color:black;">
                        <strong>🤖 Assistant:</strong><br>
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("👋 Start chatting by asking a question below!")
    
    # Suggested questions
    st.markdown("**💡 Suggested Questions:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("What is the monthly rent amount?"):
            handle_suggested_question("What is the monthly rent amount?")
        if st.button("What are my responsibilities as a tenant?"):
            handle_suggested_question("What are my responsibilities as a tenant?")
    
    with col2:
        if st.button("What is the lease duration?"):
            handle_suggested_question("What is the lease duration?")
        if st.button("What happens if I want to terminate early?"):
            handle_suggested_question("What happens if I want to terminate early?")
    
    # Chat input
    user_question = st.chat_input("Type your question here...")
    
    if user_question:
        handle_user_question(user_question)

def handle_suggested_question(question: str):
    """Handle suggested question clicks"""
    if not st.session_state.get("general_qa_retriever"):
        st.error("❌ General Q&A RAG retriever not initialized")
        return
    # Add user message to history for display
    st.session_state.messages.append({"role": "user", "content": question})
    # Extract past messages for function
    past_messages_for_llm = [
        {"role":msg["role"], "content":msg["content"]} 
        for msg in st.session_state.messages
        if msg["role"] in ("user", "assistant") and msg["content"]
    ]
    
    # Implement chatbot response
    with st.spinner("Thinking..."):
        try:
            response = answer_contextual_question_openai(
                user_question=question,
                general_qa_retriever=st.session_state.general_qa_retriever,
                ta_report=st.session_state.verification_results,
                past_messages=past_messages_for_llm
            )
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"❌ Chatbot response failed: {str(e)}")
            st.session_state.messages.pop()  # Remove last user message on failure
    st.rerun()

def handle_user_question(question: str):
    """Handle user question input"""
    if not st.session_state.get("general_qa_retriever"):
        st.error("❌ General Q&A RAG retriever not initialized")
        return
    # Add user message to history for display
    st.session_state.messages.append({"role": "user", "content": question})
    # Extract past messages for function
    past_messages_for_llm = [
        {"role":msg["role"], "content":msg["content"]} 
        for msg in st.session_state.messages
        if msg["role"] == ["user", "assistant"] and msg["content"]
    ]
    # Implement chatbot response
    with st.spinner("Thinking..."):
        try:
            response = answer_contextual_question_openai(
                user_question=question,
                general_qa_retriever=st.session_state.general_qa_retriever,
                ta_report=st.session_state.verification_results,
                past_messages=past_messages_for_llm
            )
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"❌ Chatbot response failed: {str(e)}")
            st.session_state.messages.pop()  # Remove last user message on failure
    st.rerun()

def create_export_section():
    """Section 5: Export Results"""
    
    st.markdown('<div class="section-header">📥 5. Export Analysis Report</div>', unsafe_allow_html=True)
    
    if not st.session_state.uploaded_file_name:
        st.info("📋 Upload and analyze a document first to enable export")
        return
    
    st.markdown("""
    <div class="result-box" style="color:black;">
    <h4>📄 Generate Comprehensive Report</h4>
    <p>Export all analysis results, chat history, and findings in a formatted report.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Code space for export functionality
    with st.expander("🔧 Export Code Space", expanded=False):
        st.code("""
# === EXPORT CODE ===
# TODO: Implement export logic here

def generate_export_report():
    '''
    Generate comprehensive analysis report
    
    Returns:
        str: Formatted report content
    '''
    
    report = f'''
    TENANCY AGREEMENT ANALYSIS REPORT
    ================================
    
    Document: {st.session_state.uploaded_file_name}
    Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    
    1. RAG VERIFICATION RESULTS
    ---------------------------
    {st.session_state.rag_results}
    
    2. CONTRACT ANALYSIS
    -------------------
    {st.session_state.verification_results}
    
    3. CHAT HISTORY
    --------------
    {st.session_state.messages}
    
    '''
    
    return report

# Generate preview
if st.button("Generate Preview"):
    report = generate_export_report()
    st.session_state.export_preview = report
        """, language="python")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write("**Export Options:**")
        
        export_format = st.selectbox(
            "Select format:",
            ["PDF Report", "Text File (.txt)", "JSON Data", "Markdown (.md)"]
        )
        
        include_chat = st.checkbox("Include chat history", value=True)
        include_analysis = st.checkbox("Include AI analysis", value=True)
        include_rag = st.checkbox("Include RAG verification", value=True)
    
    with col2:
        st.write("**Actions:**")
        
        if st.button("🔍 Preview", type="secondary"):
            with st.spinner("Generating preview..."):
                # TODO: Implement actual preview generation
                st.session_state.export_preview = f"""
# TENANCY AGREEMENT ANALYSIS REPORT

**Document:** {st.session_state.uploaded_file_name}
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## Summary
This is a placeholder preview. Export functionality to be implemented.

## RAG Verification
- Status: {st.session_state.rag_results if st.session_state.rag_results else 'Not run'}

## Contract Analysis
- Status: {st.session_state.verification_results if st.session_state.verification_results else 'Not run'}

## Chat History
- Messages: {st.session_state.messages if st.session_state.messages else 'Not run'}

---
*Generated by LeaseOwl*
                """
                st.success("✅ Preview generated!")
    
    # Display preview
    if st.session_state.export_preview:
        st.markdown("---")
        st.markdown("**📄 Export Preview:**")
        
        with st.container():
            st.markdown(st.session_state.export_preview)
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("✅ Confirm & Export", type="primary", use_container_width=True):
                # TODO: Implement actual export
                st.download_button(
                    label="📥 Download Report",
                    data=st.session_state.export_preview,
                    file_name=f"tenancy_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
                st.success("✅ Report ready for download!")
                st.balloons()

def main():
    """Main application function"""
    
    # Initialize
    setup_custom_css()
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🦉 LeaseOwl</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered Tenancy Agreement Analyzer & Chat Assistant</p>', unsafe_allow_html=True)
    
    # Sidebar
    create_sidebar()
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("⚠️ Please configure your OpenAI API Key in the sidebar")
    
    # Main content
    st.markdown("---")
    
    # Section 1: Upload
    create_upload_section()

    st.markdown("---")
    
    # Section 2: Translation
    create_translation_section()
    
    st.markdown("---")
    
    # Section 3: Contract Verification
    create_contract_verification_section()
    
    st.markdown("---")
    
    # Section 4: Chatbot
    create_chat_section()

    st.markdown("---")

    # Section 5: Export
    create_export_section()
   
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p><strong>LeaseOwl</strong> - Your Tenancy Agreement Assistant</p>
        <p>Built with ❤️ using Streamlit & OpenAI</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()