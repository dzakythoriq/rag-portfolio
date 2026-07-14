import os
import sys
from pathlib import Path
from dotenv import load_dotenv

import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

load_dotenv()

PDF_DIR       = Path("data/pdfs")
CHROMA_DIR    = Path("data/chroma_db")
COLLECTION    = "rag_docs"
EMBED_MODEL   = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 50

def setup_models():
    """Initialize embedding model and LLM."""
    print("Loading embedding model (first run will download)...")
    
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    
    llm = Groq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    # Set global settings in LlamaIndex
    Settings.embed_model = embed_model
    Settings.llm = llm
    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP
    
    print("Models loaded.")
    return embed_model, llm


def load_documents():
    """Load All PDF from directory."""
    if not PDF_DIR.exists() or not list(PDF_DIR.glob("*.pdf")):
        print(f"ERROR: No PDF files found in {PDF_DIR}")
        sys.exit(1)
    
    print(f"Loading PDFs from {PDF_DIR}...")
    docs = SimpleDirectoryReader(
        input_dir=str(PDF_DIR),
        required_exts=[".pdf"]
    ).load_data()
    
    print(f"Loaded {len(docs)} pages from {len(set(d.metadata.get('file_name') for d in docs))} files.")
    return docs


def build_index(docs):
    """Chunk documents, embed, and save to ChromaDB."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Setup ChromaDB
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    
    # Remove existing collection if any (to avoid duplicates during re-run)
    existing = [c.name for c in chroma_client.list_collections()]
    if COLLECTION in existing:
        print(f"Collection '{COLLECTION}' already exists, removing and rebuilding...")
        chroma_client.delete_collection(COLLECTION)
    
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION)
    vector_store      = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context   = StorageContext.from_defaults(vector_store=vector_store)
    
    # Chunking
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    print("Chunking and embedding documents (this is the longest step)...")
    index = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        transformations=[splitter],
        show_progress=True
    )
    
    print(f"\nDone! Index saved to {CHROMA_DIR}")
    return index


def main():
    setup_models()
    docs  = load_documents()
    index = build_index(docs)
    
    # Quick sanity check
    print("\nSanity check - testing simple query...")
    retriever = index.as_retriever(similarity_top_k=3)
    nodes     = retriever.retrieve("What is RAG?")
    
    print(f"Retrieved {len(nodes)} chunks. First sample chunk:")
    print("-" * 50)
    print(nodes[0].text[:300])
    print("-" * 50)
    print("\nIngestion completed. Proceed to query pipeline.")


if __name__ == "__main__":
    main()