import os
import json
from pathlib import Path
from dotenv import load_dotenv

import chromadb
from groq import Groq as GroqClient
from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.schema import NodeWithScore, TextNode
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

load_dotenv()

# ── CONFIG ──────────────────────────────────────────────
CHROMA_DIR      = Path("data/chroma_db_advanced")
COLLECTION      = "rag_docs_advanced"
EMBED_MODEL     = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_RETRIEVAL = 20   # ambil banyak dulu untuk hybrid search
TOP_K_RERANK    = 5    # setelah reranking, ambil 5 terbaik
# ────────────────────────────────────────────────────────

QA_PROMPT = PromptTemplate(
    "You are an assistant that answers questions strictly based on the provided context.\n"
    "If the answer is not in the context, say 'I cannot find this information in the provided documents.'\n"
    "Do not make up information.\n\n"
    "Context:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "Question: {query_str}\n"
    "Answer: "
)


# ── STEP 1: Load Index ───────────────────────────────────
def load_advanced_index():
    """Load the Advanced RAG index from ChromaDB."""
    chroma_client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
    chroma_collection = chroma_client.get_collection(COLLECTION)
    vector_store      = ChromaVectorStore(chroma_collection=chroma_collection)

    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    llm         = Groq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY")
    )

    Settings.embed_model = embed_model
    Settings.llm         = llm

    index = VectorStoreIndex.from_vector_store(vector_store)
    return index, embed_model


# ── STEP 2: HyDE ─────────────────────────────────────────
def hyde_transform(question: str) -> str:
    """
    Generate a hypothetical answer from the user's question.
    This answer will be embedded for search — not the original query.

    Why: user queries are short and not very descriptive.
    Hypothetical answers are longer and the language
    similar to academic documents → more informative embeddings.
    """
    client = GroqClient(api_key=os.getenv("GROQ_API_KEY"))

    prompt = (
        "You are an expert in AI, machine learning, and information retrieval systems. "
        "Your knowledge covers RAG (Retrieval-Augmented Generation), vector databases, "
        "LLMs, and related NLP techniques.\n\n"
        "Generate a detailed technical paragraph from an AI/ML research paper "
        "that would answer the following question. "
        "Be specific about AI and NLP concepts. "
        "Do not answer about any other domain.\n\n"
        f"Question: {question}\n\n"
        "Technical paragraph from AI research paper:"
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.3   # sedikit creative tapi tidak terlalu random
    )

    hypothetical = response.choices[0].message.content.strip()
    return hypothetical


# ── STEP 3: Hybrid Search ────────────────────────────────
def hybrid_search(index, embed_model, question: str, hypothetical: str) -> list:
    """
    Combine BM25 (keyword) + Vector Search (semantic).
    Vector search uses hypothetical document from HyDE.
    BM25 uses original query — because BM25 looks for exact keywords.

    Why different queries:
    - HyDE for vector: make embeddings more informative
    - Original query for BM25: technical terms remain searchable
    """

    # ── Vector Search with HyDE ────────────────────────
    # Embed jawaban hipotetikal (bukan query asli)
    hyde_embedding = embed_model.get_text_embedding(hypothetical)

    # Query ChromaDB langsung dengan vektor hipotetikal
    chroma_client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
    chroma_collection = chroma_client.get_collection(COLLECTION)

    vector_results = chroma_collection.query(
        query_embeddings=[hyde_embedding],
        n_results=TOP_K_RETRIEVAL,
        include=["documents", "metadatas", "distances"]
    )

    # Ekstrak chunks dari hasil vector search
    vector_chunks = []
    for i, doc in enumerate(vector_results["documents"][0]):
        vector_chunks.append({
            "text":     doc,
            "metadata": vector_results["metadatas"][0][i],
            "score":    1 - vector_results["distances"][0][i],  # convert distance ke similarity
            "source":   "vector"
        })

    # ── BM25 Search with original query ───────────────────
    # Get all chunks from ChromaDB for BM25 index
    all_docs = chroma_collection.get(include=["documents", "metadatas"])

    # Tokenize all chunks for BM25
    # BM25 requires a list of list of words
    tokenized_corpus = [doc.lower().split() for doc in all_docs["documents"]]
    bm25             = BM25Okapi(tokenized_corpus)

    # Search with original query (not hypothetical)
    tokenized_query = question.lower().split()
    bm25_scores     = bm25.get_scores(tokenized_query)

    # Get top-K from BM25
    import numpy as np
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:TOP_K_RETRIEVAL]

    bm25_chunks = []
    for idx in top_bm25_indices:
        if bm25_scores[idx] > 0:  # skip chunk with score 0 (no keyword match)
            bm25_chunks.append({
                "text":     all_docs["documents"][idx],
                "metadata": all_docs["metadatas"][idx],
                "score":    float(bm25_scores[idx]),
                "source":   "bm25"
            })

    # ── Reciprocal Rank Fusion ───────────────────────────
    # Combine vector and BM25 results using RRF
    # RRF formula: score = 1 / (rank + k) for each list
    # k=60 is the standard constant used in the original RRF paper
    merged = reciprocal_rank_fusion(vector_chunks, bm25_chunks, k=60)

    return merged


def reciprocal_rank_fusion(list1: list, list2: list, k: int = 60) -> list:
    """
    Combine two ranked lists into one using RRF.

    Formula: score(chunk) = sum(1 / (rank + k)) for each list
    where the chunk appears.

    Chunks that appear in both lists get higher scores
    because they get contributions from two different sources.
    """
    scores = {}

    # Scores from list 1 (vector search)
    for rank, chunk in enumerate(list1):
        key = chunk["text"][:100]  # use first 100 chars as unique key
        if key not in scores:
            scores[key] = {"chunk": chunk, "rrf_score": 0}
        scores[key]["rrf_score"] += 1 / (rank + k)

    # Scores from list 2 (BM25) — chunks appearing in both get additional scores
    for rank, chunk in enumerate(list2):
        key = chunk["text"][:100]
        if key not in scores:
            scores[key] = {"chunk": chunk, "rrf_score": 0}
        scores[key]["rrf_score"] += 1 / (rank + k)

    # Sort based on RRF score, take top-K
    sorted_chunks = sorted(
        scores.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )[:TOP_K_RETRIEVAL]

    return [item["chunk"] for item in sorted_chunks]


# ── STEP 4: Reranking ────────────────────────────────────
def rerank(question: str, chunks: list) -> list:
    """
    Use Cross-Encoder to rerank hybrid search results.

    Cross-Encoder evaluates (query, chunk) TOGETHER — more accurate
    than cosine similarity which evaluates them separately.

    Input  : top-20 chunks from hybrid search
    Output : top-5 chunks really most relevant
    """
    print(f"  Reranking {len(chunks)} chunks → ambil top {TOP_K_RERANK}...")

    reranker = CrossEncoder(RERANKER_MODEL)

    # Buat pasangan (query, chunk_text) untuk setiap chunk
    pairs = [(question, chunk["text"]) for chunk in chunks]

    # Cross-Encoder score setiap pasangan
    scores = reranker.predict(pairs)

    # Gabungkan chunk dengan scorenya, sort descending
    ranked = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )

    # Ambil top-K terbaik
    top_chunks = [chunk for chunk, score in ranked[:TOP_K_RERANK]]

    print(f"  Top reranker scores: {[round(s, 3) for _, s in ranked[:TOP_K_RERANK]]}")
    return top_chunks


# ── STEP 5: Generate Jawaban ─────────────────────────────
def generate_answer(question: str, chunks: list) -> str:
    """
    Generate final answer from top chunks after reranking.
    Same as Naive RAG, but chunks are much higher quality.
    """
    context = "\n\n---\n\n".join([chunk["text"] for chunk in chunks])

    client = GroqClient(api_key=os.getenv("GROQ_API_KEY"))

    prompt = (
        "You are an assistant that answers questions strictly based on the provided context.\n"
        "If the answer is not in the context, say 'I cannot find this information in the provided documents.'\n"
        "Do not make up information.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0
    )

    return response.choices[0].message.content.strip()


# ── MAIN PIPELINE ─────────────────────────────────────────
def query_advanced(question: str, index=None, embed_model=None) -> dict:
    """
    Full Advanced RAG pipeline.
    index and embed_model can be passed from outside (for API)
    or loaded by itself (for direct testing).
    """
    if index is None or embed_model is None:
        index, embed_model = load_advanced_index()

    print(f"\n{'='*60}")
    print(f"Query: {question}")
    print(f"{'='*60}")

    # Step 1: HyDE
    print("\n[1/4] HyDE: generating hypothetical document...")
    hypothetical = hyde_transform(question)
    print(f"  Hypothetical (preview): {hypothetical[:150]}...")

    # Step 2: Hybrid Search
    print("\n[2/4] Hybrid Search: BM25 + Vector with RRF...")
    chunks = hybrid_search(index, embed_model, question, hypothetical)
    print(f"  Retrieved {len(chunks)} chunks after RRF")

    # Step 3: Reranking
    print("\n[3/4] Reranking with Cross-Encoder...")
    top_chunks = rerank(question, chunks)

    # Step 4: Generate
    print("\n[4/4] Generating answer...")
    answer = generate_answer(question, top_chunks)

    return {
        "answer":       answer,
        "hypothetical": hypothetical,
        "sources": [
            {
                "text":  chunk["text"][:400],
                "file":  chunk["metadata"].get("file_name", "unknown"),
                "score": chunk.get("score")
            }
            for chunk in top_chunks
        ]
    }


# ── Quick test ────────────────────────────────────────────
if __name__ == "__main__":
    test_questions = [
        "What is Retrieval-Augmented Generation?",
        "What are the main limitations of Naive RAG?",
        "How does reranking improve RAG performance?",
    ]

    for q in test_questions:
        result = query_advanced(q)
        print(f"\nAnswer: {result['answer']}")
        print(f"Sources: {[s['file'] for s in result['sources']]}")
        print("-" * 60)