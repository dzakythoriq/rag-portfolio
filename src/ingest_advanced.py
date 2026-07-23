import os
import sys
from pathlib import Path
from dotenv import load_dotenv

import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

load_dotenv()

PDF_DIR        = Path("data/pdfs")
CHROMA_DIR     = Path("data/chroma_db_advanced")  # folder berbeda dari Naive
COLLECTION     = "rag_docs_advanced"               # nama berbeda dari Naive
EMBED_MODEL    = "BAAI/bge-small-en-v1.5"

# Semantic Chunking config
BREAKPOINT_PERCENTILE = 95  # makin tinggi = chunk makin besar, makin sedikit potongan

def setup_models():
    """Initialize embedding model dan LLM."""
    print("Loading embedding model...")
    
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    
    llm = Groq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    Settings.embed_model = embed_model
    Settings.llm         = llm
    
    print("Models loaded.")
    return embed_model, llm


def load_documents():
    """Load all PDFs from the directory."""
    if not PDF_DIR.exists() or not list(PDF_DIR.glob("*.pdf")):
        print(f"ERROR: No PDFs found in {PDF_DIR}")
        sys.exit(1)
    
    print(f"Loading PDFs from {PDF_DIR}...")
    docs = SimpleDirectoryReader(
        input_dir=str(PDF_DIR),
        required_exts=[".pdf"]
    ).load_data()
    
    print(f"Loaded {len(docs)} pages from {len(set(d.metadata.get('file_name') for d in docs))} files.")
    return docs


def build_advanced_index(docs, embed_model):
    """
    Semantic chunking + save to a new ChromaDB.
    Different from Naive: the splitter is SemanticSplitterNodeParser,
    not fixed-size SentenceSplitter.
    """
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Setup ChromaDB — folder dan collection DIFFERENT dari Naive
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    
    existing = [c.name for c in chroma_client.list_collections()]
    if COLLECTION in existing:
        print(f"Collection '{COLLECTION}' already exists, deleting and rebuilding...")
        chroma_client.delete_collection(COLLECTION)
    
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION)
    vector_store      = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context   = StorageContext.from_defaults(vector_store=vector_store)
    
    # ── INI PERBEDAAN UTAMA DARI NAIVE RAG ──────────────
    # Naive pakai: SentenceSplitter(chunk_size=512, chunk_overlap=50)
    # Advanced pakai: SemanticSplitterNodeParser
    #
    # SemanticSplitterNodeParser cara kerjanya:
    # 1. Embed setiap kalimat satu-satu
    # 2. Hitung "jarak" semantik antara kalimat yang berurutan
    # 3. Kalau jaraknya melebihi threshold (breakpoint_percentile) → potong di situ
    # 4. Kalau jaraknya kecil (topik masih sama) → gabung dalam satu chunk
    #
    # Hasilnya: setiap chunk berisi satu konsep yang utuh,
    # tidak terpotong di tengah penjelasan
    # ────────────────────────────────────────────────────
    splitter = SemanticSplitterNodeParser(
        embed_model=embed_model,
        breakpoint_percentile_threshold=BREAKPOINT_PERCENTILE
    )
    
    print("Chunking with Semantic Splitter and document embedding...")
    print("(Slower than Naive because each sentence is embedded first)")
    
    index = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        transformations=[splitter],
        show_progress=True
    )
    
    print(f"\nDone! Advanced index saved to {CHROMA_DIR}")
    return index


def compare_chunking_stats(index):
    """
    Show chunking statistics to compare with Naive RAG.
    In Naive RAG, all chunks are exactly 512 tokens.
    In Advanced, chunk size varies depending on semantic boundaries.
    """
    print("\n── Chunking Statistics ──────────────────────────")
    
    retriever = index.as_retriever(similarity_top_k=10)
    nodes     = retriever.retrieve("What is RAG?")
    
    chunk_lengths = [len(n.text.split()) for n in nodes]
    
    print(f"Sample 10 chunks retrieved:")
    print(f"  Average chunk length: {sum(chunk_lengths)/len(chunk_lengths):.0f} words")
    print(f"  Shortest chunk: {min(chunk_lengths)} words")
    print(f"  Longest chunk: {max(chunk_lengths)} words")
    print(f"\nNaive RAG: all chunks ~380 words (fixed)")
    print(f"Advanced : chunks vary according to semantic boundaries")
    print("─────────────────────────────────────────────────")


def main():
    embed_model, _ = setup_models()
    docs           = load_documents()
    index          = build_advanced_index(docs, embed_model)
    
    compare_chunking_stats(index)
    
    # Sanity check
    print("\nSanity check...")
    retriever = index.as_retriever(similarity_top_k=3)
    nodes     = retriever.retrieve("What is HyDE?")
    
    print(f"Retrieved {len(nodes)} chunks for query 'What is HyDE?'")
    print("Sample first chunk:")
    print("-" * 50)
    print(nodes[0].text[:400])
    print("-" * 50)
    print("\nAdvanced ingestion completed. Continue to pipeline_advanced.py")


if __name__ == "__main__":
    main()