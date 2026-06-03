"""
Medical RAG Question Answering — Hugging Face Spaces demo.

A self-contained version of the notebook pipeline:
  load PubMedQA -> chunk -> embed -> FAISS index -> retrieve -> grounded generation.

Designed to run on a FREE CPU Space. The FAISS index is built once at startup and
cached to disk, so subsequent restarts are fast. flan-t5-base runs on CPU here
(a few seconds per question) — fine for a demo.

Author: Kushagra Aggarwal
"""

import os
import pickle

import numpy as np
import torch
import faiss
import gradio as gr
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-base"
CACHE_DIR = "cache"
INDEX_PATH = os.path.join(CACHE_DIR, "faiss.index")
META_PATH = os.path.join(CACHE_DIR, "corpus.pkl")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

YESNOMAYBE_INSTR = (
    "You are a biomedical question-answering assistant. "
    "Using ONLY the context, answer the question with exactly one word: yes, no, or maybe."
)

print(f"[startup] device = {DEVICE}")

# ----------------------------------------------------------------------------
# Data + index (built once, then cached to disk)
# ----------------------------------------------------------------------------
def chunk_text(text, chunk_size=80, overlap=20):
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + chunk_size]))
        start += chunk_size - overlap
    return chunks


def flatten(example, idx):
    ctx = example["context"]
    if isinstance(ctx, dict):
        sentences = ctx.get("contexts") or ctx.get("abstract") or []
        if isinstance(sentences, str):
            sentences = [sentences]
        passage = " ".join(sentences)
    else:
        passage = str(ctx)
    return {"doc_id": idx, "passage": passage,
            "label": example["final_decision"].strip().lower()}


def build_corpus():
    """Load PubMedQA and build the chunk-level corpus + metadata."""
    try:
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled")["train"]
    except Exception as e:
        print("[startup] primary repo failed, using bigbio mirror:", e)
        ds = load_dataset("bigbio/pubmed_qa",
                          "pubmed_qa_labeled_fold0_source")["train"]

    chunks, meta = [], []
    for i in range(len(ds)):
        row = flatten(ds[i], i)
        if not row["passage"]:
            continue
        for ci, ch in enumerate(chunk_text(row["passage"])):
            chunks.append(ch)
            meta.append({"doc_id": row["doc_id"], "label": row["label"], "chunk_id": ci})
    return chunks, meta


print("[startup] loading embedder ...")
embedder = SentenceTransformer(EMB_MODEL, device=DEVICE)

if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
    print("[startup] loading cached FAISS index ...")
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        cached = pickle.load(f)
    corpus_chunks, chunk_meta = cached["chunks"], cached["meta"]
else:
    print("[startup] building corpus + FAISS index (first run only) ...")
    corpus_chunks, chunk_meta = build_corpus()
    emb = embedder.encode(corpus_chunks, batch_size=64, convert_to_numpy=True,
                          normalize_embeddings=True, show_progress_bar=True)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb.astype("float32"))
    os.makedirs(CACHE_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump({"chunks": corpus_chunks, "meta": chunk_meta}, f)

print(f"[startup] index ready: {index.ntotal} vectors over {len(corpus_chunks)} chunks")

# ----------------------------------------------------------------------------
# Generator
# ----------------------------------------------------------------------------
print("[startup] loading generator ...")
gen_tok = AutoTokenizer.from_pretrained(GEN_MODEL)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL).to(DEVICE)
gen_model.eval()


def retrieve(query, k=4):
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idxs = index.search(q_emb.astype("float32"), k)
    return [{"text": corpus_chunks[i], "meta": chunk_meta[i], "score": float(s)}
            for s, i in zip(scores[0], idxs[0])]


def generate(question, context, max_new_tokens=8):
    prompt = f"{YESNOMAYBE_INSTR}\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
    inputs = gen_tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = gen_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return gen_tok.decode(out[0], skip_special_tokens=True).strip()


def rag_answer(question, k=4):
    hits = retrieve(question, k=k)
    context = " ".join(h["text"] for h in hits)
    return generate(question, context), hits


# ----------------------------------------------------------------------------
# Gradio UI
# ----------------------------------------------------------------------------
def demo_fn(question, k):
    if not question or not question.strip():
        return "Please enter a question.", ""
    answer, hits = rag_answer(question, k=int(k))
    evidence = "\n\n".join(
        f"[score {h['score']:.3f} | doc {h['meta']['doc_id']}] {h['text']}" for h in hits
    )
    return answer, evidence


EXAMPLES = [
    ["Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?", 4],
    ["Does preoperative statin therapy reduce atrial fibrillation after coronary artery bypass grafting?", 4],
    ["Is the promise of specialty pharmaceuticals worth the price?", 4],
]

with gr.Blocks(title="Medical RAG QA") as demo:
    gr.Markdown(
        "# Medical RAG Question Answering\n"
        "Retrieval-Augmented Generation over **1,000 PubMed abstracts**. "
        "Ask a biomedical yes/no/maybe question — the system retrieves supporting "
        "evidence with FAISS and generates a grounded answer.\n\n"
        "*Built with sentence-transformers + FAISS + flan-t5-base. "
        "Answers are model-generated and not medical advice.*"
    )
    with gr.Row():
        q_in = gr.Textbox(label="Biomedical question", lines=2,
                          placeholder="e.g. Does vitamin D supplementation reduce risk of respiratory infection?")
        k_in = gr.Slider(1, 8, value=4, step=1, label="Documents to retrieve (k)")
    btn = gr.Button("Ask", variant="primary")
    ans_out = gr.Textbox(label="Grounded answer")
    ev_out = gr.Textbox(label="Retrieved evidence", lines=10)
    gr.Examples(EXAMPLES, inputs=[q_in, k_in])
    btn.click(demo_fn, inputs=[q_in, k_in], outputs=[ans_out, ev_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
