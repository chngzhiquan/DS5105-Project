# 🏡 Tenancy Agreement Checker Chatbot (TA Checker)

A **Generative AI-powered assistant** designed to help tenants review and understand their **tenancy agreements (TAs)** — ensuring transparency, fairness, and peace of mind before signing a lease.

---

## 📘 Project Overview

Residential tenancy agreements are often filled with **unclear terms and hidden clauses**, leaving tenants confused or at risk of unfair disputes.  
**TA Checker** automatically analyzes tenancy contracts, compares them to industry standards, and highlights potential issues — all through an interactive chatbot interface.

### 👤 User Persona

**Richard Chen**, a 28-year-old tech professional new to Singapore, faced challenges like:
- Spending 6+ hours researching unclear clauses  
- Unexpected repair and cleaning costs  
- Missing “Quiet Enjoyment” clauses causing disruptions  

TA Checker empowers users like Richard to sign fair agreements confidently.

---

## 💡 Key Features

| Feature | Description |
|----------|--------------|
| **Clause Comparison** | Cross-references user-uploaded TA against CEA standard templates |
| **Risk Highlighting** | Flags ambiguous, missing, or non-standard clauses |
| **Explainable AI** | Provides clear explanations and references to official clauses |
| **Interactive Q&A** | Users can chat with the bot to clarify legal terms or contract items |
| **Automated Report Generation** | Produces a quality report summarizing findings and recommendations |

---

## ⚙️ System Architecture

### 🧠 Core Components
- **Data Sources**:  
  - CEA standard TA templates and property agent guidelines
- **Preprocessing**:  
  - Document parsing, chunking, and embedding generation
- **Knowledge Bases**:  
  - `RAG 1`: TA best practices for clause critique  
  - `RAG 2`: Legal Q&A database for general tenant questions  
- **Vector Store**:  
  - FAISS index for semantic retrieval
- **Orchestration**:  
  - Built using **LangChain** for the RAG pipeline and chatbot flow
- **LLM Backbone**:  
  - OpenAI **GPT-4o-mini**
- **Frontend (MVP)**:  
  - Simple UI using **Streamlit**

---

## 🧩 Workflow

```text
Upload Tenancy Agreement (PDF/DOCX)
       ↓
File Processing → Clause Extraction & Classification
       ↓
RAG 1 Retrieval → Compare Clauses vs CEA Template
       ↓
Quality Report Generation (GPT-4o-mini)
       ↓
RAG 2 Retrieval → Q&A Support
       ↓
User Feedback → Continuous Improvement
