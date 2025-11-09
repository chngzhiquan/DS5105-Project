# 🏡 LeaseOwl [TA Checker — AI Tenancy Agreement Clause Analyzer + Chatbot]
### *Your AI legal co-pilot for fair tenancy agreements*

---

## 📘 Project Overview

Residential tenancy agreements are often filled with **unclear terms and hidden clauses**, leaving tenants confused or vulnerable to unfair terms.  

**TA Checker** automatically analyzes tenancy contracts, compares them against legal standards, and highlights potential risks — all through an interactive chatbot interface.  

This system is built on a **dual Retrieval-Augmented Generation (RAG)** architecture for precise clause analysis and context-aware Q&A.

---

## 👤 User Persona

**Richard Chen**, a 28-year-old tech professional new to Singapore, faced challenges such as:
- Spending **6+ hours** researching unclear clauses  
- Unexpected **repair and cleaning costs**  
- Missing **“Quiet Enjoyment”** clauses causing disruptions  

**TA Checker** helps users like Richard understand their contracts and sign confidently — in minutes, not hours.

---

## 💡🚀 Key Features

| **Feature** | **Description** |
|-------------|----------------|
| **Clause Comparison (RAG 1)** | Performs a deep, clause-by-clause comparison between the uploaded TA and an “ideal” clause database. |
| **Fast Whole-Doc Mode** | Provides a quick compliance check against a standard CSV checklist for rapid insights. |
| **Risk Highlighting** | Flags ambiguous, missing, or potentially unfair clauses with risk levels and suggested revisions. |
| **Explainable AI** | Each critique includes clear reasoning and references to model clauses for transparency. |
| **Interactive Q&A (RAG 2)** | Users can ask general legal questions or document-specific ones, e.g., “Why was my deposit clause flagged?” |
| **Automated Report Generation** | Summarizes key findings, risks, and suggestions into a professional review report. |
| **Multi-Language Support** | Tenancy Agreement can be automatically translated into multiple languages (Indonesian, Chinese, France, Spanish, German).

---

## ⚙️ System Architecture

**TA Checker** is powered by a **dual-RAG** framework for specialized processing and higher accuracy.

### 🔹 RAG 1: Clause-by-Clause Critique
- **Source:** Directory of “ideal” Tenancy Agreement PDFs (`./TA_template`)  
- **Process:** Splits uploaded TAs into individual clauses → queries a FAISS vector database of ideal clauses.  
- **Output:** Generates detailed clause critiques including **risk level**, **feedback**, and **suggested edits**.

### 🔹 RAG 2: Context-Aware Chatbot
- **Source:** `Database Requirements.xlsx` (Q&A pairs and legal commentary)  
- **Process:** Uses vector retrieval to find relevant answers and integrates the user’s analysis report as context.  
- **Output:** Provides both **general tenancy guidance** and **document-specific clarifications**.

### 🔹 Fast (Whole-Doc) Mode
- **Source:** CSV checklist (`./checklist/checklist.csv`)  
- **Process:** Skips retrieval; compares the entire document against the checklist for high-level compliance.  

---

## 💻 Tech Stack

| **Layer** | **Technology** |
|-----------|----------------|
| **Frontend** | Streamlit |
| **Backend & Orchestration** | LangChain |
| **LLM Engine** | OpenAI (e.g., `gpt-4o-mini`) |
| **Embeddings** | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector Store** | FAISS (Facebook AI Similarity Search) |
| **Data Handling** | Pandas, PyPDFLoader |
| **Environment** | Python, `.env` with `OPENAI_API_KEY` |

---

## 🚀 Setup and Installation

Follow this quick guide to run **TA Checker** locally.

---

### ⚙️ **1. Environment Setup**

**→ Clone the repository**
git clone https://github.com/chngzhiquan/DS5105-Project.git
cd DS5105-Project

**→ Create & activate virtual environment**
- macOS / Linux
python3 -m venv venv
source venv/bin/activate

- Windows
python -m venv venv
.\venv\Scripts\activate

**→ Install dependencies**
pip install -r requirements.txt

### 🧩 **2. Configure Keys & Files**
**→ Add your OpenAI API key**
Create a .env file in the project root:
OPENAI_API_KEY="sk-your-secret-openai-api-key"

**→ Prepare required folders & files**
| **Component**         | **Folder / File**            | **Purpose**              |
| --------------------- | ---------------------------- | ------------------------ |
| RAG 1 (Ideal Clauses) | `./TA_template/`             | Store ideal tenancy PDFs |
| RAG 2 (Chatbot Q&A)   | `Database Requirements.xlsx` | Legal Q&A source         |
| Fast Mode             | `./checklist/checklist.csv`  | Compliance checklist     |

**📁 Folder Structure Example:**
```
project_root/
├── TA_template/
│   ├── ideal_clause_1.pdf
│   └── ideal_clause_2.pdf
├── checklist/
│   └── checklist.csv (For "Fast" mode)
├── Database Requirements.xlsx (For general Q&A RAG 2)
├── main2.py
├── backend_utils_v2.py
└── .env
```

### 🧠 **3. Build Vector Databases**
**→ Run the notebook**
jupyter notebook test_rag.ipynb

**This will create:**
./faiss_index_ideal_clauses/
./faiss_index_general_qa/

*_Run again only if your data updates._*

### ▶️ **4. Run the App**
streamlit run main_v2.py

**✅ Using the App**
- Upload your Tenancy Agreement PDF.
- Click Process Document.
- Select analysis engine → RAG 1 (clause-level) or Whole-Doc.
- Click Analyze Contract.
- View your report & chat with the AI Assistant.

**📊 Example Outputs**
- Clause-level feedback with risk ratings
- Summary report highlighting missing or unfair clauses
- Chatbot Q&A referencing both general legal knowledge and your uploaded contract
- Multi-language tenancy agreement translation

**🛠️ Whole-Doc Mode**
- Compares the entire TA with a compliance checklist.
- Automatically compresses and summarizes large documents.
- Returns structured JSON and multi-language Markdown reports.

**🧩 Multi-Language Support**
- Tenancy Agreement can be translated automatically.
- Default languages: ["English","Indonesian","Chinese","French","Spanish","German, Thai, Traditional Chinese, Simplified Chinese"]

**📌 Notes**
- Ensure .env has a valid OPENAI_API_KEY.
- FAISS indexes must be built before running RAG queries.
- Whole-Doc mode works independently of RAG indexes.

**💡 Future Enhancements**
- Integration with local legal databases (e.g., CEA Singapore)
- Enhanced visual reports with clause heatmaps
- Improved OCR for scanned PDFs

**⚡ References**
LangChain Documentation
FAISS Vector Store
OpenAI GPT API

**🧠 Credits**
Developed as part of the DS5105 Project
Built with ❤️ using Streamlit, LangChain, and OpenAI GPT models

