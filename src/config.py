"""
config.py - central settings for MediGuide.

Every tunable value lives here so the rest of the code has no "magic
numbers". Values can be overridden with environment variables (or a .env
file), which is the standard way to keep secrets such as API keys out of
source code.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a .env file in the project root (if present).
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# ---- Paths -----------------------------------------------------------------
DATA_DIR = ROOT_DIR / "data"
KB_DIR = DATA_DIR / "knowledge_base"
RED_FLAGS_PATH = DATA_DIR / "red_flags.json"
CHROMA_DIR = DATA_DIR / "chroma_db"
PATIENT_DIR = DATA_DIR / "patients"
COLLECTION_NAME = "mediguide_kb"

# ---- Models ----------------------------------------------------------------
# langchain-google-genai reads GOOGLE_API_KEY from the environment itself.
LLM_MODEL = os.getenv("MEDIGUIDE_LLM_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("MEDIGUIDE_EMBEDDING_MODEL", "models/gemini-embedding-001")
# Backup models tried in order if the main model is overloaded (503) or unavailable.
# Comma-separated in .env: MEDIGUIDE_FALLBACK_MODELS=gemini-flash-latest,gemini-flash-lite-latest
FALLBACK_MODELS = [m.strip() for m in os.getenv(
    "MEDIGUIDE_FALLBACK_MODELS", "gemini-flash-latest,gemini-flash-lite-latest").split(",") if m.strip()]
LLM_TEMPERATURE = 0.2  # low = more faithful to the retrieved text

# ---- Chunking (two-stage ingestion) ---------------------------------------
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

# ---- Retrieval -------------------------------------------------------------
TOP_K = 4
# Chunks scoring below this relevance are treated as "not in the knowledge
# base". Tune it with eval/evaluate.py rather than guessing.
MIN_RELEVANCE = float(os.getenv("MEDIGUIDE_MIN_RELEVANCE", "0.35"))

# Number of previous chat turns passed to the model for follow-up questions.
HISTORY_TURNS = 4


class ConfigError(RuntimeError):
    """Raised when a required setting (such as the API key) is missing."""


def require_api_key() -> str:
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ConfigError(
            "GOOGLE_API_KEY is not set. Get a free key at "
            "https://aistudio.google.com/apikey and put it in a .env file "
            "(see .env.example)."
        )
    # langchain-google-genai looks for GOOGLE_API_KEY specifically.
    os.environ.setdefault("GOOGLE_API_KEY", key)
    return key