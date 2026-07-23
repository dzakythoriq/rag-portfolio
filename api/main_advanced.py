import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from src.pipeline_advanced import load_advanced_index, query_advanced

# ── Lifespan: load index SEKALI saat startup ─────────────
# This is the main difference from Naive RAG.
# In pipeline_advanced.py before, load_advanced_index() was called
# every query comes — very inefficient.
# Here, the index is loaded ONCE when FastAPI starts,
# then stored in a global variable and used repeatedly.
index        = None
embed_model  = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global index, embed_model
    print("Loading Advanced RAG index...")
    index, embed_model = load_advanced_index()
    print("Advanced RAG ready.")
    yield
    print("Shutting down.")


app = FastAPI(
    title="Advanced RAG Portfolio API",
    description="Advanced RAG — Semantic Chunking + Hybrid Search + Reranking + HyDE",
    version="2.0.0",
    lifespan=lifespan
)

# ── Schemas ──────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str

class SourceItem(BaseModel):
    text:  str
    file:  str
    score: float | None

class QueryResponse(BaseModel):
    answer:       str
    hypothetical: str        # this is extra from Naive RAG — expose HyDE result
    sources:      list[SourceItem]

# ── Endpoints ────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":       "ok",
        "version":      "advanced",
        "index_loaded": index is not None
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if index is None:
        raise HTTPException(status_code=503, detail="Index not loaded yet")

    # Pass index yang sudah di-load saat startup
    result = query_advanced(req.question, index=index, embed_model=embed_model)

    return QueryResponse(
        answer=result["answer"],
        hypothetical=result["hypothetical"],
        sources=[SourceItem(**s) for s in result["sources"]]
    )