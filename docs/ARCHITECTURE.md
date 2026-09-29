# MediGuide: Architecture, Viva Preparation and Report Outline

## 1. Problem and scope

After discharge, patients have questions about diet, activity and routine but cannot reach a clinician quickly. Generic chatbots may hallucinate or give unsafe medical advice. MediGuide answers only from a curated knowledge base and hard-refuses clinical questions.

In scope: four topics, free-text procedure entry, health-background intake, multiple prescriptions (image or PDF), cited answers, a static warning-sign list.
Out of scope (future work): multilingual support, real discharge-summary parsing, EHR integration, clinician dashboard.

## 2. Components

| Module | Responsibility | Key idea |
|---|---|---|
| `procedures.py` | Registry and free-text matcher | Alias phrases plus typo tolerance; unmatched input is reported, not guessed |
| `safety.py` | Rule-based gate | Deterministic, testable, no API call |
| `knowledge_base.py` | Chunking | Two-stage split keeps chunks within one topic |
| `ingest.py` | Embeddings and Chroma | Cosine similarity, persisted index |
| `patient.py` | Profile and storage | Conditions widen retrieval; context goes to the prompt |
| `prescriptions.py` | OCR/PDF to medicine list | Heuristic parse, patient can correct |
| `rag_chain.py` | LCEL pipeline | Retrieve, branch, generate; injectable components for tests |

## 3. The LCEL chain

```
RunnablePassthrough.assign(docs=retrieve)
  | RunnablePassthrough.assign(context=format_docs)
  | RunnablePassthrough.assign(answer=RunnableBranch(
        (no docs, fixed "not in the guide" reply),
        prompt | llm | StrOutputParser()))
```

Each step receives a dict and adds a key. The branch is the reason an empty retrieval never reaches the model.

## 4. Safety layer levels

| Level | Trigger | Response |
|---|---|---|
| emergency | Emergency phrases (chest pain, cannot breathe, stroke signs, heavy bleeding), self-harm phrases, BP 180/120 or above, sugar under 54 or 400 and above, temperature 40 C and above, pulse over 150 or under 40 | Call emergency services |
| urgent | BP 160/100 or above (or systolic under 90), sugar under 70 or 300 and above, temperature 38 C and above, pulse over 120 or under 50 | Contact doctor today |
| refer | Symptoms, diagnosis questions, medicine changes, side effects, interactions | Contact doctor; assistant only covers lifestyle and logistics |
| ok | Everything else | Goes to RAG |

Numeric parsing details (worth showing as a debugging story): a naive `\d+/\d+` pattern treated dates ("25/12/2026") and fractions ("3/4 cup") as blood pressure. The fix uses lookarounds, range validation (systolic 60-300, diastolic 30-200, systolic greater than diastolic), and requires a context word or unit for blood sugar so "sugar 2 spoons" is not read as a reading. Regression tests cover each case. The evaluation harness also caught two harmless questions being wrongly refused ("blood pressure" and "coughing" matched symptom words), which were fixed and tested.

## 5. Likely viva questions

1. **Why RAG instead of fine-tuning?** Content changes as guidelines change; RAG updates by editing a markdown file and re-ingesting. It also allows citations and avoids training on medical data.
2. **Why is the safety layer outside LangChain?** Safety must be deterministic and independent of model behaviour. Rules cannot be argued out of a refusal, are cheap, need no API, and can be tested exhaustively.
3. **What is an embedding?** A vector of numbers representing meaning; similar texts have vectors pointing in similar directions, measured by cosine similarity.
4. **Why two-stage chunking?** Header splitting keeps each chunk inside one topic (structure-aware). The recursive splitter then cuts long sections at natural boundaries with overlap so context is not lost.
5. **What does chunk overlap do?** Prevents a sentence at a boundary from losing its neighbours; costs a little extra storage.
6. **How do you stop hallucination?** Answer-only-from-excerpts prompt, low temperature, relevance threshold with a no-context fallback that skips the LLM, mandatory citations, and a refusal layer for clinical questions.
7. **What is LCEL?** LangChain Expression Language: composing Runnables with `|`. Every component shares `.invoke()`, so chains can be tested with fakes, batched or streamed.
8. **How is personalisation done?** Two ways: patient background is added to the prompt, and conditions widen the retrieval filter to related topics.
9. **Why exclude warning-sign chunks from retrieval?** The assistant must not interpret symptoms. Warning signs are shown as fixed text so nothing depends on the model.
10. **How do you choose the relevance threshold?** Use `python -m eval.evaluate --retrieval --threshold-sweep` and pick the value that balances hit-rate against irrelevant chunks.
11. **What are the limits of keyword-based safety?** Misses unusual phrasing and may over-refuse. Mitigations: conservative design, tests, evaluation set; future work is a trained classifier.
12. **Privacy?** Data stays local in git-ignored JSON. A deployment needs encryption, authentication, consent and retention policy.
13. **How do prescriptions work and what can go wrong?** OCR then heuristic line parsing. Errors are expected, so the patient reviews an editable table and the prompt states the data may be wrong. The bot never advises changing medicines.
14. **How would you evaluate the system?** Refusal accuracy, retrieval hit@k, answer faithfulness (human or LLM-judge review of grounding), and a clinician review of the knowledge base.
15. **What would you do next?** Clinician-reviewed content, semantic safety classifier, multilingual support, database with auth, discharge-summary parsing, larger evaluation set.

## 6. Report outline (map onto your six phases)

1. Introduction, problem statement, objectives, scope
2. Literature review: RAG, clinical chatbots, safety in medical LLMs
3. Requirements and system design (architecture diagram from README)
4. Knowledge base design and ingestion (sources, chunking, embeddings)
5. Safety layer design and testing (levels, thresholds, regression story)
6. RAG pipeline, personalisation and prescription module
7. User interface
8. Testing and evaluation (test suite, eval set, threshold sweep, results table)
9. Limitations, ethics and future work
10. Conclusion; appendices (prompts, knowledge-base samples, test output)

Fill the evaluation results table yourself from `python -m eval.evaluate --retrieval --threshold-sweep` once your API key is set.
