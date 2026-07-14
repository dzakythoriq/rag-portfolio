import os
from pathlib import Path
from dotenv import load_dotenv

import chromadb
from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

load_dotenv()

# ── CONFIG ──────────────────────────────────────────────
CHROMA_DIR  = Path("data/chroma_db")
COLLECTION  = "rag_docs"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K       = 5

# Explicit prompt template — don’t let the LLM make things up
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


def load_index():
    """Load index from ChromaDB."""
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
    return index


def build_query_engine(index):
    """Create query engine with custom prompt."""
    query_engine = index.as_query_engine(
        similarity_top_k=TOP_K,
        text_qa_template=QA_PROMPT,
    )
    return query_engine


def query_with_sources(query_engine, question: str) -> dict:
    """
    Query and return answer + source chunks.
    Return format:
    {
        "answer": str,
        "sources": [{"text": str, "file": str, "score": float}]
    }
    """
    response = query_engine.query(question)
    
    sources = []
    for node in response.source_nodes:
        sources.append({
            "text":  node.text[:400],   # preview 400 char
            "file":  node.metadata.get("file_name", "unknown"),
            "score": round(node.score, 4) if node.score else None
        })
    
    return {
        "answer":  str(response),
        "sources": sources
    }


# ── Quick test kalau dijalankan langsung ────────────────
if __name__ == "__main__":
    print("Loading index...")
    index        = load_index()
    query_engine = build_query_engine(index)
    
    test_questions = [
        "What is Retrieval-Augmented Generation?",
        "What are the main limitations of Naive RAG?",
        "How does hybrid search improve RAG performance?"
    ]
    
    for q in test_questions:
        print(f"\nQ: {q}")
        result = query_with_sources(query_engine, q)
        print(f"A: {result['answer']}")
        print(f"Sources: {[s['file'] for s in result['sources']]}")
        print("-" * 60)