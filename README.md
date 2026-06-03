---
title: Medical RAG QA
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Medical RAG Question Answering

A Retrieval-Augmented Generation (RAG) demo for biomedical yes/no/maybe questions,
built over 1,000 expert-annotated PubMed abstracts (PubMedQA).

**Pipeline:** sentence-transformer embeddings → FAISS vector search → evidence
retrieval → grounded generation with `flan-t5-base`.

Enter a biomedical question and the app retrieves supporting passages and returns
a grounded answer plus the evidence it used.

> Answers are model-generated for demonstration only and are **not** medical advice.

**Author:** Kushagra Aggarwal — MS Data Science, Northeastern University
