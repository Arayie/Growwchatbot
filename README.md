# Groww Facts-Only SBI Mutual Fund AI Assistant (Powered by RAG)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![SEBI Compliant](https://img.shields.io/badge/SEBI-Facts--Only-green.svg)](https://investor.sebi.gov.in/)

> A zero-hallucination, facts-only Retrieval-Augmented Generation (RAG) web application engineered to deliver instant, 100% accurate factual answers regarding SBI AMC mutual fund schemes with official citations.

---

## 🏗️ System Architecture

```text
+-----------------------------------------------------------------------+
|                       USER INTERFACE LAYER                            |
|    Streamlit Dark Mode Web App + Real-Time Web Speech Voice Agent     |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|                    INTENT CLASSIFIER & GUARDRAILS                     |
|  - PII Scrubber (PAN, Aadhaar, Phone, Folio, Email [REDACTED])        |
|  - Advisory Refusal Router ("Should I buy?" -> Factsheet Citation)    |
|  - Scope Enforcement ("Non-MF Queries" -> Polite Refusal)             |
+-----------------------------------+-----------------------------------+
                                    |
                        +-----------+-----------+
                        |                       |
                        v                       v
+-------------------------------+   +-----------------------------------+
|      STRUCTURED SQL ENGINE    |   |       UNSTRUCTURED VECTOR STORE   |
|   (SQLite: sbi_scheme_facts)  |   |    (ChromaDB: sentence-embeds)    |
|  - Exact NAV, Min SIP, TER,   |   |  - Procedural FAQs, Downloads,    |
|    Exit Loads, AUM, Benchmark |   |    ELSS Lock-in Rules, Objectives |
+---------------+---------------+   +-------------------+---------------+
                |                                       |
                +-------------------+-------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|                   RESPONSE GENERATION & FOLLOW-UP ENGINE              |
|  - Length Constraint: <= 3 Sentences                                  |
|  - Mandatory Official Source Citation Link                            |
|  - Timestamp Stamp: Last updated from sources: YYYY-MM-DD             |
|  - 3 Contextual Dynamic Follow-up Prompt Pills                        |
+-----------------------------------------------------------------------+
```

---

## 🌟 Key Features

1. **Zero Hallucination Factual Answers**:
   - Structured numerical facts (NAV, Min SIP, Expense Ratio, Exit Load, AUM, Riskometer) sourced directly from SQLite database records to prevent hallucination.
2. **SEBI Regulatory Safety & Advice Refusal**:
   - Deterministic refusal guardrail for investment advice, recommendations, or stock tips with links to official scheme factsheets.
3. **PII Protection & Sanitization**:
   - Automatic scrubbing of PAN numbers, Aadhaar numbers, phone numbers, email addresses, and folio numbers before processing.
4. **Multimodal Voice Agent**:
   - Audio voice recording integration with speech-to-text transcription.
5. **Interactive UI with Groww Dark Theme**:
   - Custom CSS theme (`#0D0F12`), starter prompt cards, source citation badges (`Source: https://...`), and clickable dynamic follow-up pills.

---

## 📁 Repository Structure

```text
SBI-mf-RAG/
├── data/                      # Raw scheme documents, PDFs, and .webloc shortcuts
├── chroma_db/                 # Embedded ChromaDB vector store (all-MiniLM-L6-v2)
├── sbi_scheme_facts.db        # SQLite database storing numerical scheme facts
├── db.py                      # Database schema and CRUD functions
├── ingest.py                  # Ingestion script (scans ./data/, parses PDFs, populates SQL & ChromaDB)
├── guardrails.py              # PII scrubber & regex sanitizer
├── router.py                  # Intent classifier (NON_MF_QUERY, ADVICE_OR_OPINION, AMBIGUOUS, FACTUAL)
├── rag_engine.py              # Unified Hybrid RAG execution engine & follow-up generator
├── app.py                     # FastAPI REST API server (/api/health, /api/chat)
├── streamlit_app.py           # Native Streamlit Web UI application
├── verify_phase1.py           # Phase 1 verification script (SQL + ChromaDB check)
├── verify_phase2.py           # Phase 2 verification script (Guardrails & Router check)
├── verify_phase3.py           # Phase 3 verification script (Hybrid RAG Engine check)
├── verify_phase4.py           # Phase 4 verification script (FastAPI Endpoints check)
├── verify_phase5.py           # Phase 5 verification script (Web UI assets check)
├── red_team_audit.py          # Phase 6 Red-teaming safety audit & latency benchmark
├── index.html                 # HTML/JS Standalone Web UI
├── DESIGN.md                  # Frontend design system documentation
├── DESIGN_PROMPTS.md          # System prompt & guardrail rules documentation
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # Project documentation
```

---

## 🚀 Quickstart & Local Setup

### 1. Clone & Setup Virtual Environment
```bash
git clone https://github.com/Arayie/SBImf-chatbot-RAG.git
cd SBImf-chatbot-RAG

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Ingestion Pipeline
```bash
python3 ingest.py
```

### 3. Launch Streamlit Web Application
```bash
streamlit run streamlit_app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser!

---

## 🧪 Running Verifications & Red-Teaming Audit

```bash
# Run Red-Teaming Safety Audit & Latency Benchmark
python3 red_team_audit.py
```

---

## ☁️ Deploying to Streamlit Community Cloud

1. Push your repository to GitHub:
   ```bash
   git add .
   git commit -m "Complete Facts-Only SBI Mutual Fund RAG Assistant"
   git push origin main
   ```
2. Log into [share.streamlit.io](https://share.streamlit.io/).
3. Click **New app**, select your repository `Arayie/SBImf-chatbot-RAG`, set main file to `streamlit_app.py`, and click **Deploy**!
