"""
MediGuide CLI
--------------
Quick command-line entry point for testing the RAG chain without the
Streamlit UI.

Examples:
    python main.py --procedure appendectomy --query "can I eat rice?"
    python main.py --procedure appendectomy --query "can I eat rice?" --no-llm
"""

import argparse

from src.rag_chain import answer_query, list_procedures


def main():
    parser = argparse.ArgumentParser(description="MediGuide — post-discharge care assistant (CLI)")
    parser.add_argument("--patient", default="demo_patient", help="Patient/session ID")
    parser.add_argument("--procedure", required=True, help="Procedure/condition key")
    parser.add_argument("--query", required=True, help="Patient's question")
    parser.add_argument("--no-llm", action="store_true", help="Dry run — no API key needed")
    args = parser.parse_args()

    available = list_procedures()
    if args.procedure not in available:
        parser.error(f"--procedure must be one of {available}")

    result = answer_query(args.patient, args.procedure, args.query, use_llm=not args.no_llm)
    print("REFUSED:", result["refused"])
    print("SOURCES:", result["sources"])
    print("ANSWER:\n", result["answer"])


if __name__ == "__main__":
    main()
