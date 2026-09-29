# MediGuide

A grounded, safety-first post-discharge patient care assistant built with LangChain LCEL, ChromaDB, and Google Gemini.

---

## Project Structure

```
mediguide/
├── app/
│   └── streamlit_app.py      # Streamlit web application
├── data/
│   ├── knowledge_base/       # Curated aftercare guidelines (.md per procedure)
│   ├── red_flags.json        # Rule-based safety gate keywords & vital sign thresholds
│   └── patients/             # Patient profile JSON store (git-ignored)
├── docs/
│   └── ARCHITECTURE.md       # Architecture details, viva preparation & report outline
├── eval/
│   ├── eval_set.json         # Evaluation queries (safety + retrieval ground truth)
│   └── evaluate.py           # Safety accuracy & retrieval hit-rate evaluation harness
├── src/
│   ├── config.py             # Central settings & configuration
│   ├── ingest.py             # Knowledge base chunking & ChromaDB vectorstore ingestion
│   ├── knowledge_base.py     # Two-stage split (Markdown headers + recursive character)
│   ├── patient.py            # Patient profile, intake, and local persistence
│   ├── prescriptions.py      # Prescription OCR & PDF text parser
│   ├── procedures.py         # Procedure registry & typo-tolerant phrase matcher
│   ├── rag_chain.py          # LangChain Expression Language (LCEL) RAG pipeline
│   └── safety.py             # Deterministic 4-level safety gate (emergency, urgent, refer, ok)
├── tests/                    # Comprehensive unit & integration test suite (115 passing tests)
├── main.py                   # CLI interface
└── requirements.txt
```

---

## Quickstart

### 1. Installation

```bash
pip install -r requirements.txt
```

> **Optional OCR dependency**: Tesseract is recommended for scanning image-based prescriptions:
> - macOS: `brew install tesseract`
> - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`

### 2. Environment Setup

Configure your Google Gemini API key:
```bash
export GOOGLE_API_KEY="your_api_key_here"
```
Or create a `.env` file in the project root:
```ini
GOOGLE_API_KEY=your_api_key_here
```

### 3. Ingest Knowledge Base

Build the ChromaDB vector database from curated aftercare documentation:
```bash
python -m src.ingest
```

### 4. Running the Application

#### Streamlit Web App
```bash
streamlit run app/streamlit_app.py
```

#### CLI Interface
```bash
# List supported procedures
python main.py --list

# Single question
python main.py --procedure appendectomy --query "can I eat rice?"

# Natural language procedure description
python main.py --describe "I had my appendix removed" --query "can I take a shower?"

# Interactive chat
python main.py --procedure c_section
```

### 5. Running Tests & Evaluation

```bash
# Run full test suite
pytest -v

# Run safety layer evaluation (offline, zero-cost)
python -m eval.evaluate

# Run retrieval evaluation & threshold sweep (requires built vectorstore)
python -m eval.evaluate --retrieval --threshold-sweep
```

