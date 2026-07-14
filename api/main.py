import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from src.pipeline import load_index, build_query_engine, query_with_sources

# ── Lifespan: Load the index once at startup ────────────
query_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global query_engine
    print("Loading RAG index...")
    index        = load_index()
    query_engine = build_query_engine(index)
    print("RAG ready.")
    yield
    print("Shutting down.")

app = FastAPI(
    title="RAG Portfolio API",
    description="Naive RAG over research papers",
    version="1.0.0",
    lifespan=lifespan
)

# ── Schemas ─────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str

class SourceItem(BaseModel):
    text:  str
    file:  str
    score: float | None

class QueryResponse(BaseModel):
    answer:  str
    sources: list[SourceItem]

# ── Endpoints ───────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "index_loaded": query_engine is not None}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    if query_engine is None:
        raise HTTPException(status_code=503, detail="Index not loaded yet")
    
    result = query_with_sources(query_engine, req.question)
    return QueryResponse(
        answer=result["answer"],
        sources=[SourceItem(**s) for s in result["sources"]]
    )