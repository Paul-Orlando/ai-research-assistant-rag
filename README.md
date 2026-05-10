# AI Research Assistant (RAG)
### Semantic Retrieval + Grounded Answer Generation — v2

A lightweight but production-minded RAG (Retrieval-Augmented
Generation) research tool built with Python and the OpenAI API.
Retrieves relevant context, generates grounded answers, and detects
potential hallucinations via cosine similarity grounding score.

---

## Key Features

- Semantic retrieval via OpenAI text embeddings
- Grounded answer generation — answers only from retrieved context
- Hallucination detection with score interpretation (High/Medium/Low)
- Embedding cache — avoids redundant API calls across runs
- Error handling on all API calls
- Context token length guard (~750 tokens)
- JSONL interaction logging
- Vectorstore stub — ready to scale to ChromaDB, Pinecone, or FAISS

---

## Tech Stack

- Python 3.11+
- OpenAI API (gpt-4.1-mini + text-embedding-3-small)
- NumPy
- Scikit-learn (cosine similarity)

---

## Grounding Score Interpretation

| Score | Interpretation |
|---|---|
| >= 0.80 | High — answer well grounded |
| 0.60-0.79 | Medium — mostly grounded, minor risk |
| < 0.60 | Low — potential hallucination |

---

## Setup

1. Clone the repo
2. Install dependencies:
