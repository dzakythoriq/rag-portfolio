# RAG Portfolio — From Naive to Advanced

Proyek ini adalah implementasi sistem **Retrieval-Augmented Generation (RAG)** yang dibangun secara bertahap dari Naive RAG hingga Advanced RAG sebagai bagian dari portofolio AI Engineer.

Ide dasarnya sederhana: daripada hanya membuat satu aplikasi jadi, saya mendokumentasikan seluruh perjalanan pengembangan — termasuk kegagalan yang ditemukan di Naive RAG dan bagaimana setiap teknik di Advanced RAG memperbaikinya secara terukur.

---

## Fase 1 — Naive RAG (Selesai)

### Stack
| Komponen | Tools |
|---|---|
| PDF Parser | PyMuPDF |
| Embedding Model | BAAI/bge-small-en-v1.5 (lokal) |
| Vector Store | ChromaDB |
| Orchestration | LlamaIndex |
| LLM | Llama 3.3 70B via Groq API |
| Backend | FastAPI |
| Frontend | Streamlit |
| Evaluasi | RAGAS (custom implementation) |

### Arsitektur
```
PDF files
    ↓ PyMuPDF (ekstrak teks)
Raw text per halaman
    ↓ SentenceSplitter (512 token, overlap 50)
Chunks
    ↓ BAAI/bge-small-en-v1.5 (embedding)
Vectors
    ↓ ChromaDB (simpan)
Vector Database
    ↓ (saat query) cosine similarity search
Top-5 relevant chunks
    ↓ Llama 3.3 70B (generate)
Jawaban + sources
```

### Baseline Evaluation Results
Dievaluasi menggunakan 5 pertanyaan terhadap corpus 3 paper RAG dari arXiv.

| Metric | Score |
|---|---|
| Faithfulness | 0.5000 |
| Answer Relevancy | 0.8549 |
| Context Precision | 0.9600 |
| Context Recall | 0.8503 |

### Identified Failure Cases
Dari hasil evaluasi, ditemukan dua kelemahan utama Naive RAG yang menjadi motivasi pengembangan ke Advanced RAG:

**1. Faithfulness rendah (0.50)** — Dua pertanyaan tentang HyDE dan Reranking menghasilkan Faithfulness 0.0, artinya LLM menjawab di luar konteks dokumen yang di-retrieve (hallucination). Penyebabnya: topik tersebut tidak cukup tercover di chunks yang berhasil di-retrieve.

**2. Fixed-size chunking** — Chunking 512 token tanpa mempertimbangkan batas semantik menyebabkan potongan yang tidak kontekstual. Kalimat penting bisa terpotong di tengah, menurunkan kualitas retrieval.

Kedua masalah ini menjadi target perbaikan di Fase 2.

---

## Fase 2 — Advanced RAG (In Progress)

Teknik yang akan diimplementasikan berdasarkan failure cases yang ditemukan:

- **Semantic Chunking** — menggantikan fixed-size chunking
- **Hybrid Search** — kombinasi BM25 + vector search
- **Reranking** — cross-encoder untuk filter ulang hasil retrieval
- **HyDE** — query transformation sebelum retrieval

---

## Cara Menjalankan

### Prerequisites
- Python 3.10+
- Groq API key (gratis di [console.groq.com](https://console.groq.com))

### Setup

```bash
# Clone repo
git clone https://github.com/username/rag-portfolio.git
cd rag-portfolio

# Buat virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Setup environment variable
cp .env.example .env
# Edit .env dan isi GROQ_API_KEY dengan key kamu
```

### Tambahkan PDF
Taruh file PDF ke folder `data/pdfs/`. Project ini menggunakan 3 paper RAG dari arXiv sebagai corpus default:
- [RAG Original Paper](https://arxiv.org/abs/2005.11401)
- [Advanced RAG Survey](https://arxiv.org/abs/2312.10997)
- [RAG vs Fine-tuning](https://arxiv.org/abs/2401.08406)

### Jalankan

```bash
# Step 1: Ingest PDF ke ChromaDB
python src/ingest.py

# Step 2: Jalankan backend (terminal 1)
uvicorn api.main:app --reload --port 8000

# Step 3: Jalankan frontend (terminal 2)
streamlit run frontend/app.py

# Step 4 (opsional): Evaluasi
python evaluation/evaluate.py
```

Frontend tersedia di `http://localhost:8501`
API docs tersedia di `http://localhost:8000/docs`

---

## Struktur Project

```
rag-portfolio/
├── data/
│   └── pdfs/              ← PDF corpus
├── src/
│   ├── ingest.py          ← Pipeline indexing (PDF → ChromaDB)
│   └── pipeline.py        ← RAG query pipeline
├── api/
│   └── main.py            ← FastAPI backend
├── frontend/
│   └── app.py             ← Streamlit UI
├── evaluation/
│   ├── evaluate.py        ← Evaluation runner
│   └── naive_rag_results.csv
├── .env.example
├── requirements.txt
└── README.md
```

---

## Catatan Teknis

**Kenapa BAAI/bge-small-en-v1.5?**
Model embedding lokal tanpa API call — gratis, cepat, dan performa kompetitif untuk dokumen berbahasa Inggris. Dipilih sebagai baseline sebelum eksperimen dengan model yang lebih besar.

**Kenapa chunk size 512 token?**
Angka ini intentionally arbitrary untuk Naive RAG — cukup untuk konteks bermakna tapi bukan hasil optimasi. Limitasi ini yang diidentifikasi sebagai salah satu penyebab Faithfulness rendah dan akan diperbaiki dengan Semantic Chunking di Advanced RAG.

**Kenapa Groq?**
Groq menyediakan inferensi Llama yang sangat cepat dengan free tier yang cukup untuk development dan demo.
