# 🏡 LeaseOwl [TA Checker — AI Tenancy Agreement Clause Analyzer + Chatbot]
### *Your AI legal co-pilot for fair tenancy agreements*

---

## 📘 Project Overview

**TA Checker** is an AI-powered tool designed to help tenants and legal professionals **analyze tenancy agreements efficiently.** It:
Extracts, compares, and evaluates clauses against an ideal template
Flags ambiguous, missing, or unfair clauses
Provides context-aware Q&A via an AI chatbot
Generates professional summary reports
Goal: Help users understand contracts and make confident decisions in **minutes instead of hours.**

---

## 👤 User Persona

**Richard Chen**, a 28-year-old tech professional new to Singapore, faced challenges such as:
- Spending **6+ hours** researching unclear clauses  
- Unexpected **repair and cleaning costs**  
- Missing **“Quiet Enjoyment”** clauses causing disruptions  

**TA Checker** helps users like Richard understand their contracts and sign confidently — in minutes, not hours.

---

## 💡🚀 Key Features

| **Feature**                           | **Description**                                                                                                                                       |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Clause-by-Clause Critique (RAG 1)** | Thoroughly compares each clause in your TA against an “ideal” clause database; provides **risk ratings, feedback, and suggested edits**.              |
| **Fast Whole-Doc Mode**               | Quick compliance check using a **CSV checklist**, ideal for large documents or rapid insights.                                                        |
| **Risk Highlighting**                 | Flags ambiguous, missing, or potentially unfair clauses; highlights **critical vs minor issues**.                                                     |
| **Explainable AI**                    | Provides **clear reasoning** and references to model clauses for every flagged issue.                                                                 |
| **Context-Aware Chatbot (RAG 2)**     | Ask general or document-specific questions, e.g., “Why was my deposit clause flagged?”; integrates the analysis report for **relevant answers**.      |
| **Automated Report Generation**       | Summarizes **key findings, risks, and suggestions** into professional reports, ready to share or save.                                                |
| **Multi-Language Support**            | Translates TAs automatically into multiple languages (English, Indonesian, Chinese, French, Spanish, German, Thai, Traditional & Simplified Chinese). |
| **Thorough Clause Coverage**          | Supports **deep, clause-level analysis** with full risk context; goes beyond basic compliance checks.                                                 |
                                             |

---

## ⚙️ System Architecture

**TA Checker** uses a **dual-RAG** framework for accurate and context-aware analysis

### 🔹 RAG 1: Clause-by-Clause Critique
- **Source:** ./TA_template/ — ideal tenancy PDFs 
- **Process:** Splits uploaded TA into clauses → queries FAISS vector store → returns **detailed clause critique** with risk scores and suggestions.
- **Output:** Generates detailed clause critiques including **risk level**, **feedback**, and **suggested edits**.

### 🔹 RAG 2: Context-Aware Chatbot
- **Source:** `Database Requirements.xlsx` — general legal Q&A & commentary
- **Process:** Vector retrieval + user report context → answers **general and document-specific queries.**
- **Output:** Provides both **general tenancy guidance** and **document-specific clarifications**.

### 🔹 Fast (Whole-Doc) Mode
- **Source:** CSV checklist (`./checklist/checklist.csv`)  
- **Process:** Skips vector retrieval; compares the entire document against the checklist for **high-level compliance.** 

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
- Detailed clause-level critiques with risk ratings
- Summary report highlighting missing/unfair clauses
- Chatbot Q&A referencing both general knowledge and your uploaded TA
- Multi-language TA translations
- Whole-Doc mode JSON & Markdown summaries

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

