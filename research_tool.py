"""
AI Research Tool (v2)
---------------------
Features:
- Semantic retrieval (embeddings)
- Grounded answer generation
- Lightweight hallucination check with score interpretation
- Embedding cache (avoids redundant API calls)
- Error handling on all API calls
- Context token length guard
- Simple JSONL logging (gitignored)

Requirements:
    pip install openai numpy scikit-learn
    export OPENAI_API_KEY="your_api_key_here"
"""

import json
import pickle
import os
import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# ==============================
# CONFIG
# ==============================
MODEL           = "gpt-4.1-mini"
EMBED_MODEL     = "text-embedding-3-small"
LOG_FILE        = "tool_log.jsonl"       # add to .gitignore
CACHE_FILE      = "doc_embeddings.pkl"   # persists across runs
MAX_CONTEXT_CHARS = 3000                 # ~750 tokens, safe for most models
GROUNDING_HIGH  = 0.80
GROUNDING_MED   = 0.60

client = OpenAI()

# ==============================
# KNOWLEDGE BASE (replaceable)
# ==============================
# To scale: swap with ChromaDB, Pinecone, FAISS, or a file loader
# documents = load_from_vectorstore()
documents = [
    "Black holes form from gravitational collapse of massive stars.",
    "Thermohaline circulation regulates global ocean heat transport.",
    "Quantum mechanics governs microscopic physical systems.",
]

# ==============================
# EMBEDDING FUNCTION
# ==============================
def embed(text: str) -> np.ndarray | None:
    """Return embedding vector for text, or None on failure."""
    try:
        response = client.embeddings.create(model=EMBED_MODEL, input=text)
        return np.array(response.data[0].embedding)
    except Exception as e:
        print(f"[Embedding error] {e}")
        return None

# ==============================
# EMBEDDING CACHE
# ==============================
def load_or_build_embeddings(docs: list[str]) -> np.ndarray | None:
    """Load cached embeddings from disk or build and cache them."""
    if os.path.exists(CACHE_FILE):
        print("[Cache] Loading precomputed embeddings...")
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)

    print("[Cache] Building embeddings (one-time)...")
    embeddings = []
    for doc in docs:
        emb = embed(doc)
        if emb is None:
            print(f"[Warning] Failed to embed document: '{doc[:60]}...'")
            return None
        embeddings.append(emb)

    result = np.array(embeddings)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(result, f)
    print("[Cache] Embeddings saved.\n")
    return result

doc_embeddings = load_or_build_embeddings(documents)

# ==============================
# RETRIEVAL
# ==============================
def retrieve(query: str, k: int = 2) -> list[str]:
    """Return top-k most relevant documents for the query."""
    if doc_embeddings is None:
        return []

    q_emb = embed(query)
    if q_emb is None:
        return []

    sims = cosine_similarity([q_emb], doc_embeddings)[0]
    idx  = sims.argsort()[-k:][::-1]
    return [documents[i] for i in idx]

# ==============================
# CONTEXT GUARD
# ==============================
def safe_context(docs: list[str]) -> str:
    """Join docs and truncate to MAX_CONTEXT_CHARS to avoid token overflow."""
    context = "\n".join(docs)
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS]
        print(f"[Warning] Context truncated to {MAX_CONTEXT_CHARS} characters.")
    return context

# ==============================
# GENERATION (GROUNDED)
# ==============================
def generate_answer(query: str, context: str) -> str | None:
    """Generate a grounded answer using only the provided context."""
    prompt = f"""Answer the question using ONLY the provided context.
If the context does not contain enough information, say so clearly.

Context:
{context}

Question:
{query}
"""
    try:
        response = client.responses.create(
            model=MODEL,
            input=prompt,
            temperature=0.5,
            max_output_tokens=300,
        )
        return response.output_text
    except Exception as e:
        print(f"[Generation error] {e}")
        return None

# ==============================
# GROUNDING CHECK
# ==============================
def grounding_score(answer: str, context: str) -> float | None:
    """Cosine similarity between answer and context embeddings."""
    a_emb = embed(answer)
    c_emb = embed(context)
    if a_emb is None or c_emb is None:
        return None
    return float(cosine_similarity([a_emb], [c_emb])[0][0])

def interpret_score(score: float) -> str:
    """Human-readable grounding interpretation."""
    if score >= GROUNDING_HIGH:
        return "High — answer is well grounded in context"
    elif score >= GROUNDING_MED:
        return "Medium — answer is mostly grounded, minor risk"
    else:
        return "Low — potential hallucination, treat with caution"

# ==============================
# LOGGING
# ==============================
def log_interaction(query, context_docs, answer, score):
    """Append interaction record to JSONL log file."""
    record = {
        "timestamp":       str(datetime.datetime.utcnow()),
        "query":           query,
        "context":         context_docs,
        "answer":          answer,
        "grounding_score": score,
    }
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        print(f"[Logging error] {e}")

# ==============================
# MAIN TOOL LOOP
# ==============================
def run_tool():
    if doc_embeddings is None:
        print("[Error] Could not build document embeddings. Exiting.")
        return

    print("AI Research Tool v2  |  type 'exit' to quit\n")

    while True:
        query = input("User: ").strip()
        if not query:
            continue
        if query.lower() == "exit":
            print("Goodbye.")
            break

        # Retrieve
        retrieved_docs = retrieve(query)
        if not retrieved_docs:
            print("[Error] Retrieval failed. Check your API key and try again.\n")
            continue

        context = safe_context(retrieved_docs)

        # Generate
        answer = generate_answer(query, context)
        if not answer:
            print("[Error] Generation failed.\n")
            continue

        print(f"\nAnswer:\n{answer}")

        # Grounding
        score = grounding_score(answer, context)
        if score is not None:
            label = interpret_score(score)
            print(f"\nGrounding Score: {round(score, 3)}  |  {label}")
        else:
            print("\nGrounding Score: unavailable")

        print()

        # Log
        log_interaction(query, retrieved_docs, answer, score)

# ==============================
# ENTRY POINT
# ==============================
if __name__ == "__main__":
    run_tool()