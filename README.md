# SafeX Solutions — FAQ Chatbot (AI/ML Prototype)

A retrieval-augmented FAQ chatbot for [safexsolutions.com](https://safexsolutions.com),
built as an internship prototype at SafeX Solutions.

## Architecture

```
User -> React frontend -> FastAPI backend -> sentence-embedding retriever -> Groq (Llama 3.3 70B) -> answer
```

1. **Knowledge base** (`backend/knowledge_base.py`) - 13 factual chunks scraped
   and manually curated from SafeX's real Home, Services, and Contact pages
   (about, mission, each service line, contact info, Trust, etc.).
2. **Retriever** (`backend/retriever.py`) - dense sentence embeddings
   (`all-MiniLM-L6-v2` from sentence-transformers) + cosine similarity over
   the knowledge base. Embeddings capture semantic meaning rather than
   literal word overlap, so paraphrased questions ("how do I reach your
   team?") still match the right chunk even without shared keywords.
   Embeddings are computed once at startup and cached in memory - no
   external vector DB needed at this scale (~13 documents).
3. **Backend** (`backend/main.py`) - FastAPI service. On each chat request:
   - retrieves top-3 relevant chunks for the user's question
   - sends them as grounding context to Groq with a system
     prompt that forces the model to answer only from that context
     (prevents hallucinated services/prices/claims)
   - returns the answer plus the retrieved sources and match scores, so the
     retrieval step is auditable
4. **Evaluation** (`backend/evaluate_retriever.py`) - a hand-labeled set of
   15 test questions (each paired with the knowledge-base chunk it should
   retrieve) used to measure retrieval quality offline, reporting Hit Rate
   @1, Hit Rate @3, and Mean Reciprocal Rank (MRR).
5. **Frontend** (`frontend/`) - React + Vite + Tailwind CSS, styled as a
   security-console session log rather than a generic chat widget (fitting
   for a cybersecurity/IT company). Each answer shows a vertical confidence
   meter per retrieved source - height and color encode match strength -
   making retrieval transparency the visual signature of the UI rather than
   a decorative afterthought.

## How to Run it

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the URL Vite prints (default `http://localhost:5173`). The dev server
proxies `/api/*` to `http://localhost:8000`, so both must be running.

### Health check

`GET http://localhost:8000/api/health` reports whether the Groq API key is
configured and how many knowledge-base entries are loaded.

### Evaluating the retriever

```bash
cd backend
python evaluate_retriever.py
```

Prints per-question retrieval results plus aggregate Hit Rate @1, Hit Rate
@3, and MRR - useful for checking retrieval quality after editing the
knowledge base or swapping the embedding model.

## Why this design

- **Grounded, not generic**: the system prompt explicitly forbids the model
  from inventing services or prices - if the retriever finds nothing
  relevant, the bot points the user to `contact@safexsolutions.com` /
  `+92 327 5781580` instead of guessing.
- **Semantic retrieval, still lightweight**: sentence embeddings give much
  better recall on paraphrased questions than pure keyword matching, while
  still avoiding the need for an external vector database at this scale
  (~13 documents fit comfortably in memory).
- **Measured, not just demoed**: the evaluation script gives a concrete,
  reproducible signal for retrieval quality (Hit Rate / MRR) rather than
  relying on "it seemed to work when I tried it."
- **Extensible**: to grow this into a full product, swap `knowledge_base.py`
  for a scraper/CMS sync job, and swap the in-memory embedding matrix for a
  proper vector store (e.g. pgvector or FAISS) once the KB grows past a few
  hundred chunks.

## Extending the knowledge base

Add entries to `KNOWLEDGE_BASE` in `backend/knowledge_base.py` — each is a
dict with `id`, `topic`, and `text`. No retraining needed; TF-IDF re-fits at
startup.
