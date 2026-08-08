"""
Tiny, dependency-free stemmer used consistently at both indexing and
query time, so word forms like "shower"/"showers" or "walk"/"walking"
match each other in TF-IDF even though it's a purely lexical method.
Avoids needing NLTK's downloadable data, keeping the pipeline fully
offline-runnable. Deliberately simple (suffix stripping) — a fine,
explainable choice for the beta stage.
"""

import re

SUFFIXES = ["ing", "edly", "ed", "es", "s"]


def simple_stem(word: str) -> str:
    word = word.lower()
    for suf in SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


def tokenize_and_stem(text: str):
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return [simple_stem(w) for w in words]


_RAW_STOP_WORDS = [
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "i", "you", "he", "she", "it", "we", "they", "my", "your", "his",
    "her", "its", "our", "their", "this", "that", "these", "those",
    "and", "or", "but", "if", "so", "of", "to", "in", "on", "at", "for",
    "with", "about", "can", "could", "should", "would", "will", "shall",
    "do", "does", "did", "have", "has", "had", "what", "when", "where",
    "how", "which", "who", "whom",
]
STEMMED_STOP_WORDS = sorted(set(simple_stem(w) for w in _RAW_STOP_WORDS))
