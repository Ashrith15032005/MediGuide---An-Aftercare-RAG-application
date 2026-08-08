# MediGuide (Beta)

A Retrieval-Augmented Generation (RAG) based post-discharge patient care assistant.

## Project Structure

```
mediguide/
├── app/
│   └── streamlit_app.py      # Demo UI
├── data/
│   ├── knowledge_base/       # Curated aftercare docs (one .md per procedure)
│   └── red_flags.json        # Safety layer keyword patterns
├── src/
│   ├── config.py             # Paths, model settings, prompts
│   ├── ingest.py             # Builds the TF-IDF retrieval index
│   ├── rag_chain.py          # Retrieval + generation orchestrator
│   ├── safety.py             # Rule-based refusal layer
│   ├── patient_history.py    # Per-patient session context
│   ├── prescription_ocr.py   # Prescription scan (OCR + parsing)
│   └── text_utils.py         # Shared stemmer/tokenizer
├── tests/
│   └── test_safety.py
├── main.py                   # CLI entry point
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Tesseract OCR is required for the prescription scan feature:
- Mac: `brew install tesseract`
- Ubuntu/Debian: `sudo apt install tesseract-ocr`
- Windows: https://github.com/UB-Mannheim/tesseract/wiki

Set your Anthropic API key for live answers (optional — dry-run mode works without it):
```bash
export ANTHROPIC_API_KEY=your_key_here
```

## Build the retrieval index

Run once, and again whenever you edit `data/knowledge_base/*.md`:
```bash
python -m src.ingest
```

## Run

```bash
# CLI
python main.py --procedure appendectomy --query "can I eat rice?"
python main.py --procedure appendectomy --query "can I eat rice?" --no-llm   # no API key needed

# Streamlit app
streamlit run app/streamlit_app.py

# Tests
pytest tests/ -v
```

## Architecture

```
Patient selects procedure -> asks a question
        |
        v
   Safety Layer (rule-based symptom detection, data/red_flags.json)
    |                          |
 refuse                    proceed
    |                          v
    |                 Retrieval (TF-IDF + stemming, procedure-scoped)
    |                          v
    |                 Patient History + Prescription context
    |                          v
    |                 LLM generation (grounded, must cite section)
    |                          v
    +------------------> Answer shown to patient (with sources)
```

## Known Limitations (Beta) / Future Work

- Retrieval uses TF-IDF, not neural embeddings — swap in `sentence-transformers` or an API embedding model for semantic search
- Prescription OCR uses heuristic line parsing — works best on printed prescriptions; handwriting is noisy
- No multilingual support yet
- Patient history is single-session, file-based — needs a real database + auth for multi-user deployment
- Safety layer is keyword-based — a trained classifier would catch more nuanced symptom phrasing

## Evaluation (Recommended Next Step)

Build a test set of ~20-30 questions per procedure with known-correct answers, and measure retrieval accuracy, answer correctness, and refusal accuracy.
