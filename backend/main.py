"""
SafeX Solutions FAQ Chatbot — FastAPI backend.

Architecture (small RAG pipeline):
  1. User question comes in via POST /api/chat
  2. Sentence-embedding retriever (all-MiniLM-L6-v2) pulls the most
     relevant knowledge-base chunks by semantic similarity
  3. Retrieved chunks + question are sent to Groq (Llama 3.3 70B)
     with a system prompt instructing it to answer ONLY from context
  4. Response (with the chunks that were used) is returned to the client

Run:
  export GROQ_API_KEY=gsk_...
  uvicorn main:app --reload --port 8000
"""

import os
import time
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from groq import Groq

from retriever import KnowledgeRetriever, RetrievedChunk

load_dotenv()  # reads backend/.env into os.environ, if present

app = FastAPI(title="SafeX FAQ Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = KnowledgeRetriever()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

MODEL_NAME = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are the official FAQ assistant for SafeX Solutions \
(safexsolutions.com), a global tech, cybersecurity, marketing, and creative \
media services company.

Rules:
- Answer ONLY using the CONTEXT provided below. Do not invent services, \
prices, offices, or claims that aren't in the context.
- If the context doesn't contain the answer, say you don't have that \
information and suggest the user contact SafeX directly at \
contact@safexsolutions.com or +92 327 5781580.
- Be concise, friendly, and professional — 2-4 sentences unless the user \
asks for a list.
- Never mention "context", "knowledge base", or "retrieval" to the user; \
just answer naturally as SafeX's assistant.
"""


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    history: List[ChatMessage] = Field(default_factory=list)


class SourceChunk(BaseModel):
    id: str
    topic: str
    score: float


class ChatResponse(BaseModel):
    reply: str
    sources: List[SourceChunk]
    latency_ms: int


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "llm_configured": client is not None,
        "kb_size": len(retriever.kb),
    }


@app.get("/api/suggested-questions")
def suggested_questions():
    return {
        "questions": [
            "What services does SafeX Solutions offer?",
            "How can I contact SafeX Solutions?",
            "What is the SafeX Trust and how does it work?",
            "Does SafeX offer cybersecurity services?",
            "What is the Skill Development Centre?",
            "Which countries does SafeX operate in?",
        ]
    }


def build_context_block(chunks: List[RetrievedChunk]) -> str:
    if not chunks:
        return "No relevant information found in the knowledge base."
    parts = []
    for c in chunks:
        parts.append(f"[{c.topic}]\n{c.text}")
    return "\n\n".join(parts)


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if client is None:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured on the server.",
        )

    start = time.time()
    chunks = retriever.retrieve(req.message, top_k=3)
    context_block = build_context_block(chunks)

    messages = []
    for m in req.history[-6:]:  # keep last few turns for light continuity
        messages.append({"role": m.role, "content": m.content})

    messages.append(
        {
            "role": "user",
            "content": f"CONTEXT:\n{context_block}\n\nQUESTION: {req.message}",
        }
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=400,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *messages,
            ],
        )
        reply_text = response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

    latency_ms = int((time.time() - start) * 1000)

    return ChatResponse(
        reply=reply_text,
        sources=[
            SourceChunk(id=c.id, topic=c.topic, score=round(c.score, 3))
            for c in chunks
        ],
        latency_ms=latency_ms,
    )
