import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="RAG Portfolio",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG over Research Papers")
st.caption("Naive RAG — ChromaDB + LlamaIndex + Groq (Llama 3.1 70B)")

# ── Sidebar info ────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.write("Corpus: 3 RAG research papers from arXiv")
    st.write("Embedding: BAAI/bge-small-en-v1.5")
    st.write("LLM: Llama 3.1 70B via Groq")
    st.write("Vector Store: ChromaDB")
    st.divider()
    st.write("**Example questions:**")
    examples = [
        "What is Retrieval-Augmented Generation?",
        "What are limitations of Naive RAG?",
        "How does reranking improve retrieval quality?",
        "What is HyDE and when should it be used?"
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["question"] = ex

# ── Main UI ─────────────────────────────────────────────
question = st.text_input(
    "Ask a question about RAG:",
    value=st.session_state.get("question", ""),
    placeholder="e.g. What are the main components of a RAG pipeline?"
)

if st.button("Ask", type="primary") and question:
    with st.spinner("Retrieving and generating..."):
        try:
            resp = requests.post(
                f"{API_URL}/query",
                json={"question": question},
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            
            # Jawaban
            st.subheader("Answer")
            st.write(data["answer"])
            
            # Source chunks — ini penting untuk transparansi
            st.subheader(f"Retrieved Sources ({len(data['sources'])} chunks)")
            for i, src in enumerate(data["sources"], 1):
                with st.expander(f"Source {i} — {src['file']} (score: {src['score']})"):
                    st.write(src["text"])
                    
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Make sure FastAPI is running on port 8000.")
        except Exception as e:
            st.error(f"Error: {str(e)}")