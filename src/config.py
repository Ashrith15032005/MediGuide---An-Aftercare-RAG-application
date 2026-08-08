"""
Central configuration for MediGuide.
All paths are resolved relative to the project root so the app works
regardless of which directory you run it from.
"""

import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KB_DIR = os.path.join(ROOT_DIR, "data", "knowledge_base")
RED_FLAGS_PATH = os.path.join(ROOT_DIR, "data", "red_flags.json")
INDEX_PATH = os.path.join(ROOT_DIR, "data", "index.pkl")
SESSIONS_DIR = os.path.join(ROOT_DIR, "data", "sessions")

os.makedirs(SESSIONS_DIR, exist_ok=True)

# LLM settings
LLM_MODEL = "claude-sonnet-4-6"
LLM_MAX_TOKENS = 500

# Retrieval settings
TOP_K = 3
MIN_SCORE = 0.03

# Supported procedures (must match filenames in data/knowledge_base/*.md)
PROCEDURE_DISPLAY_NAMES = {
    "appendectomy": "Appendectomy",
    "c_section": "C-Section",
    "diabetes_hypertension": "Diabetes / Hypertension Management",
}

SYSTEM_PROMPT = """You are MediGuide, a post-discharge patient care assistant.

STRICT RULES:
- Answer ONLY using the information in the provided CONTEXT below. Do not use outside medical knowledge.
- If the context does not contain enough information to answer, say so explicitly instead of guessing.
- Always mention which section your answer is grounded in (e.g. "Based on the Diet section...").
- Keep answers short, warm, and easy to understand for a patient at home.
- Never diagnose, assess symptoms, or comment on whether something is dangerous — that is handled separately.
- If patient context (procedure, prescription, past questions) is provided, use it to personalize your answer where relevant.
"""
