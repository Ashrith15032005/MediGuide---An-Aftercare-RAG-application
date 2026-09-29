"""
MediGuide CLI - test the assistant without the Streamlit UI.

Examples (run one at a time):

    python main.py --list
    python main.py --describe "I had my appendix removed" --query "can I eat rice?"
    python main.py --procedure c_section          # interactive chat
    python main.py --procedure cardiac_recovery --patient asha --query "when can I drive?"

If a profile was saved for --patient (by the Streamlit app), it is used for
personalised answers.
"""
import argparse
import sys

from src.patient import PatientStore
from src.procedures import PROCEDURES, label_for, match_procedure


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MediGuide - post-discharge care assistant (CLI)")
    p.add_argument("--list", action="store_true", help="list supported procedures and exit")
    p.add_argument("--patient", default="demo_patient", help="patient/session id")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--procedure", choices=list(PROCEDURES), help="procedure key")
    g.add_argument("--describe", help="describe the procedure in your own words")
    p.add_argument("--query", help="ask one question; omit for interactive chat")
    return p


def resolve_procedure(args, saved_profile):
    if args.procedure:
        return args.procedure
    if args.describe:
        match = match_procedure(args.describe)
        if match is None:
            sys.exit("Could not match that description to a supported procedure. "
                     f"Supported: {', '.join(PROCEDURES)}")
        return match.key
    if saved_profile and saved_profile.procedure_key:
        return saved_profile.procedure_key
    sys.exit("Give --procedure or --describe (or save a profile in the app first).")


def show(answer) -> None:
    tag = f"[{answer.safety_level.upper()}] " if answer.refused else ""
    print(f"\n{tag}{answer.text}\n")
    for s in answer.sources:
        print(f"  {s.label}: {s.procedure} > {s.section} (relevance {s.score})")


def main() -> None:
    args = build_parser().parse_args()
    if args.list:
        for key in PROCEDURES:
            print(f"{key:24s} {label_for(key)}")
        return

    loaded = PatientStore().load(args.patient)
    profile, medicines = loaded if loaded else (None, [])
    procedure = resolve_procedure(args, profile)
    print(f"Procedure: {label_for(procedure)}")

    from src.rag_chain import MediGuideAssistant  # imported late so --list needs no key
    assistant = MediGuideAssistant()
    history: list = []

    def ask(q: str) -> None:
        ans = assistant.answer(q, procedure, profile=profile, medicines=medicines, history=history)
        show(ans)
        history.extend([("user", q), ("assistant", ans.text)])

    if args.query:
        ask(args.query)
        return
    print("Type a question, or 'quit' to exit. Not a substitute for medical advice.")
    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"quit", "exit", "q"}:
            break
        if q:
            ask(q)


if __name__ == "__main__":
    main()
