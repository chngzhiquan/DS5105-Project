import streamlit as st
import asyncio
import re
import io
import os
import tempfile
from datetime import datetime
from backend_utils_v2 import (
    ideal_clauses_retriever,
    general_qa_retriever,
    process_pdf_to_clauses,
    initialize_qa_resources,
    answer_contextual_question_openai,
    run_full_analysis,
    translate_document
)

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

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

    if "original_file_bytes" not in st.session_state:
        st.session_state.original_file_bytes = None
    
    if "uploaded_file_content" not in st.session_state:
        st.session_state.uploaded_file_content = None

    if 'parsed_clauses' not in st.session_state:
        st.session_state.parsed_clauses = None
    
    # Default language
    if "target_language" not in st.session_state:
        st.session_state.target_language = "English"
    
    # RAG state
    # Ideal Clauses RAG retriever object (RAG 1)
    if "ideal_clauses_retriever" not in st.session_state:
        st.session_state.ideal_clauses_retriever = ideal_clauses_retriever

    # General Q&A RAG retriever object (RAG 2)
    if "general_qa_retriever" not in st.session_state:
        st.session_state.general_qa_retriever = general_qa_retriever
    
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
        
        ## Logo
        st.image("Logo.jpeg", width="stretch")
        st.markdown("---")
        
        # About us
        with st.expander("🦉 About Us", expanded=False):
            st.markdown(
                """
                <div style="text-align: center;">
                    <h3 style="margin-bottom: 0;">LeaseOwl</h3>
                    <p style="font-size: 15px; margin-top: 2px; color: #666;">Know Your Lease</p>
                </div>
                <hr style="margin: 4px 0;">
                <p style="font-size: 14px; line-height: 1.5; text-align: justify;">
                    A <b>GenAI-powered assistant</b> designed to automate the review of tenancy agreements, 
                    providing tenants with instant clarity, actionable feedback, and on-demand contextual answers 
                    — so you can sign with confidence.
                </p>
                """,
                unsafe_allow_html=True
            )

        # Help Section
        with st.expander("ℹ️ How to Use", expanded=False):
            st.markdown(
                """
                ### 🪜 Steps:
                #### In Sidebar
                - **Enter** your OpenAI API key
                #### In Main Page
                1. **Upload** tenancy agreement PDF  
                2. **Translate** to your language of choice  
                3. **Check** AI-powered contract analysis  
                4. **Chat** with the Chatbot  
                5. **Export** the results  

                ### ⚙️ Requirements:
                - OpenAI API key  
                - Your PDF Tenancy Agreement
                """,
            )

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

        # Checklist path
        default_checklist = os.path.expanduser("./checklist/checklist.csv") 
        checklist_path = st.text_input(
            "Checklist file (.csv)",
            value=default_checklist,
            help="Used to compare checklist against",
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
                st.session_state.original_file_bytes = None
                st.session_state.uploaded_file_content = None
                st.session_state.parsed_clauses = None
                st.session_state.target_language = "English"
                st.session_state.verification_results = None
                st.session_state.messages = []
                st.session_state.export_preview = None
                st.rerun()
        
        
        # About
        st.markdown(
            """
            <div style='text-align: center;'>
                <p style='font-size: 14px; color: grey; margin-bottom: 0;'>
                    <b>LeaseOwl v1.0</b>
                </p>
                <p style='font-size: 13px; margin-top: 2px; margin-bottom: 0;'>
                    Your Tenancy Agreement Assistant
                </p>
                <p style='font-size: 12px; color: #999; margin-top: 2px;'>
                    © 2025 All rights reserved
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

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
        file_bytes = uploaded_file.getvalue()
        st.session_state.original_file_bytes = file_bytes
        
        # 1. Create an in-memory buffer from the bytes
        file_buffer = io.BytesIO(file_bytes)
        
        # 2. Pass the BUFFER (not bytes) to your backend function
        result = process_pdf_to_clauses(file_buffer)
        
        if result:
            # 3. Unpack the two return values
            raw_text, clauses = result
            
            # 4. Save results to session state
            st.session_state.uploaded_file_content = raw_text # This is now the extracted text
            st.session_state.parsed_clauses = clauses 
            return True
        else:
            st.error("Error: Failed to process PDF (backend returned None).")
            return False
        
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

    col1, col2 = st.columns([3,1])
    
    with col1:
        # Language selection    
        target_language = st.selectbox(
            "Select Language:",
            ["English", "Indonesian", "Thai", "Traditional Chinese", "Simplified Chinese", "Spanish", "French", "German"],
            index = 0,
            key="target_language"  
        )
    
    with col2:
        # Translate button inside function
        if st.button("🌍 Translate Document", type="primary"):
            uploaded_content = st.session_state.get('original_file_bytes')
            if not uploaded_content:
                st.error("❌ No document content available for translation")
            else:
                with st.spinner(f"Translating document to {st.session_state.target_language}..."):
                    try:
                        mapped_language = st.session_state.target_language

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
        with st.expander(f"📄 View Translated Document ({st.session_state.target_language})"):
            st.text_area(f"Translated_document", translated_text, height=400, label_visibility="collapsed")
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
        st.subheader("Choose Analysis Engine")
        analysis_mode = st.radio(
            "Choose engine:",
            ["Fast","Thorough"],
            captions=["Compares against checklist for a quick overview.","Compares against legal database for a detailed report."],
            index=0,
            key="analysis_mode_ratio",
            label_visibility="collapsed"
            )
        st.session_state.analysis_mode = analysis_mode
      
    with col2:
        if st.button("▶️ Analyze Contract", type="primary"):
            uploaded_content = st.session_state.get('uploaded_file_content')
            if not uploaded_content:
                st.error("❌ No document content available for analysis")
                return
            with st.spinner("Analyzing contract... (This may take a moment)"):
                try:
                    # Get the necessary data from session state
                    clauses = st.session_state.parsed_clauses
                    raw_text = st.session_state.uploaded_file_content
                    
                    if not raw_text:
                        st.error("❌ Failed to read text from the stored file.")
                        return

                    # Get analysis config from session state
                    target_language = st.session_state.get("target_language", "English")
                    analysis_mode_str = st.session_state.get("analysis_mode", "Fast")
                    checklist_path = st.session_state.get("checklist_path", "./checklist/checklist.csv")
                    
                    if not os.path.exists(checklist_path):
                        st.error(f"Checklist file not found at: {checklist_path}")
                        return

                    # 5. Determine mode for LangGraph
                    mode = "fast" if str(analysis_mode_str).startswith("Fast") else "thorough"

                    # 6. Run Full Analysis (from utils.py)
                    # We pass in the data we just retrieved from session state
                    report_md = asyncio.run(run_full_analysis(
                        raw_text=raw_text,
                        clauses=clauses,
                        checklist_path=checklist_path,
                        target_language=target_language,
                        analysis_mode=mode
                    ))
                    
                    # 7. Save the final report
                    st.session_state.verification_results = report_md
                    st.success(f"✅ Contract analysis completed! (Mode: {mode})")
                
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.session_state.verification_results = None
    
    # Display results if available
    if st.session_state.verification_results:
        with st.expander("📄 View Analysis Results", expanded=True):
            st.markdown(st.session_state.verification_results, unsafe_allow_html=True)

def create_chat_section():
    """Section 4: Chatbot"""
    
    st.markdown('<div class="section-header">💬 4. Chatbot</div>', unsafe_allow_html=True)
    
    if not st.session_state.uploaded_file_name:
        st.info("📋 Upload a document first to enable chat")
        return
 
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
    target_language = st.session_state.get("target_language","English")
    # Implement chatbot response
    with st.spinner("Thinking..."):
        try:
            response = answer_contextual_question_openai(
                user_question=question,
                general_qa_retriever=st.session_state.general_qa_retriever,
                ta_report=st.session_state.verification_results,
                past_messages=past_messages_for_llm,
                target_language=target_language
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
    target_language = st.session_state.get("target_language","English")
    # Implement chatbot response
    with st.spinner("Thinking..."):
        try:
            response = answer_contextual_question_openai(
                user_question=question,
                general_qa_retriever=st.session_state.general_qa_retriever,
                ta_report=st.session_state.verification_results,
                past_messages=past_messages_for_llm,
                target_language=target_language
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
   
    col1, col2 = st.columns([3, 1])
    
    with col2:     
        if st.button("🔍 Preview Results", type="secondary"):
            with st.spinner("Generating preview..."):
                chat_history_md = ""
                if 'messages' in st.session_state and st.session_state.messages:
                    chat_lines = []
                    for msg in st.session_state.messages:
                        role = msg['role'].capitalize()
                        # Use regex to remove any HTML tags like <span>
                        content = re.sub(r'<[^>]+>', '', msg['content'])
                        # Format as "Role: Content"
                        chat_lines.append(f"**{role}:** {content}\n")
                    chat_history_md = "\n".join(chat_lines)
                else:
                    chat_history_md = "No chat history found."
                
                # 2. Get Analysis Results
                analysis_results_md = st.session_state.get("verification_results", "Analysis not yet run.")

                # 3. Build the clean preview string
                st.session_state.export_preview = f"""
**Document:** {st.session_state.uploaded_file_name}
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## Contract Analysis
{analysis_results_md}

---

## Chat History
{chat_history_md}

---
*Generated by LeaseOwl*
                """
                st.success("✅ Preview generated!")
    
    # Display preview
    if st.session_state.export_preview:
        
        with st.expander("📄 View Export Preview", expanded=False):
            st.markdown(st.session_state.export_preview)
             
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("✅ Confirm & Export", type="primary", width="stretch"):
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
        <p>Built with ❤️ DSS5105 Data Science Projects in Practice</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()