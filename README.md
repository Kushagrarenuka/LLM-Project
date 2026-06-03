# Medical RAG Question Answering

An end-to-end **Retrieval-Augmented Generation (RAG)** system for biomedical question
answering over PubMed abstracts — built not just to generate answers, but to **measure
how grounded those answers are** in retrieved evidence.

**[▶ Try the live demo on Hugging Face Spaces](https://huggingface.co/spaces/kushagra1707/medical-rag)**

---

## Overview

Large language models answer from parametric memory, which makes them prone to
**hallucination** — a serious problem for medical questions. RAG addresses this by
retrieving real source documents and constraining the model to answer only from that
evidence.

This project implements a full RAG pipeline over the **PubMedQA** corpus (1,000
expert-annotated yes/no/maybe questions, each paired with a PubMed abstract) and
evaluates it against a no-retrieval baseline to quantify the benefit of retrieval.

## Results

Evaluated on a held-out set of questions (baseline = same generator, no retrieved context):

| Metric | Baseline (no RAG) | RAG |
|---|---|---|
| Accuracy | 0.45 | **0.56** |
| Macro-F1 | 0.28 | **0.34** |
| Retrieval hit-rate@4 | — | **0.99** |

Retrieval improved answer accuracy by **~11 points** over the baseline, with the correct
source document retrieved for **~99%** of questions — confirming the gain comes from
retrieval rather than the model alone.

## Architecture

```
Question
   │
   ▼
[ Embed query ]  ──►  [ FAISS vector search ]  ──►  top-k evidence chunks
   │                   (all-MiniLM-L6-v2)
   ▼
[ flan-t5-base ]  ◄── question + retrieved evidence (grounded prompt)
   │
   ▼
Grounded yes/no/maybe answer  +  the evidence used
```

**Pipeline stages**
1. **Ingest & chunk** — PubMed abstracts split into overlapping ~80-word windows for sharper retrieval.
2. **Embed & index** — chunks encoded with `sentence-transformers/all-MiniLM-L6-v2` (384-dim) and stored in a **FAISS** inner-product index (cosine similarity on normalized vectors).
3. **Retrieve** — top-k most semantically similar chunks per question.
4. **Generate** — `google/flan-t5-base` answers using only the retrieved context (deterministic greedy decoding).
5. **Evaluate** — baseline vs RAG on accuracy, macro-F1, retrieval hit-rate, and a faithfulness proxy.

## Tech Stack

`Python` · `PyTorch` · `Hugging Face Transformers` · `sentence-transformers` · `FAISS` ·
`scikit-learn` · `Gradio` · `Hugging Face Spaces`

## Repository Structure

| File | Description |
|---|---|
| `LLM_Project_Kushagra.ipynb` | Full pipeline + evaluation notebook (data → retrieval → generation → metrics → visualizations) |
| `app.py` | Standalone Gradio app powering the live demo |
| `requirements.txt` | Dependencies |
| `README.md` | This file |

## Running It

### Try the hosted demo
No setup needed — open the
**[Hugging Face Space](https://huggingface.co/spaces/kushagra1707/medical-rag)**, enter a
biomedical question, and view the grounded answer plus the evidence retrieved.

### Run the notebook (Google Colab)
1. Open `LLM_Project_Kushagra.ipynb` in Colab.
2. Set the runtime to a **GPU** (`Runtime → Change runtime type → T4 GPU`).
3. `Runtime → Run all`. The dependency install, data load, FAISS build, evaluation, and
   Gradio demo all run end to end.

### Run the app locally
```bash
pip install -r requirements.txt
python app.py
```
The FAISS index builds once on first launch and is cached to disk for fast restarts.

## Evaluation Notes

- The **`maybe`** class is rarely predicted — it is only ~9% of the data and is inherently
  ambiguous, so a small model defaults to yes/no. This is an expected limitation, not a bug.
- **Faithfulness** is approximated with a lightweight lexical (ROUGE-L) overlap between the
  answer and its evidence. It is a rough grounding signal, not a headline metric.
- **`flan-t5-base`** is intentionally small so the evaluation runs quickly; the same pipeline
  works with larger generators.

## Next Steps

- Swap in a **biomedical embedding model** (e.g. PubMedBERT-based) for stronger retrieval.
- Use **RAGAS** (faithfulness, answer-relevance, context-precision) for production-grade
  evaluation.
- Add **QLoRA fine-tuning** of the generator and extend the harness to a three-way
  baseline / RAG / fine-tuned comparison.
- Replace the flat FAISS index with **IVF/HNSW** for scalable retrieval.

## Disclaimer

Answers are model-generated for demonstration purposes only and are **not** medical advice.

---

**Author:** Kushagra Aggarwal MS Data Science, Northeastern University