# Naive RAG Baseline — July 13, 2026

## Stack
- Embedding: BAAI/bge-small-en-v1.5
- Vector Store: ChromaDB
- LLM: Llama 3.3 70B (Groq)
- Chunking: Fixed-size, 512 tokens, overlap 50
- Top-K: 5

## Scores
| Metric            | Score  |
|-------------------|--------|
| Faithfulness      | 0.5000 |
| Answer Relevancy  | 0.8549 |
| Context Precision | 0.9600 |
| Context Recall    | 0.8503 |

## Known Failure Cases
- HyDE: Faithfulness 0.0 — topic not sufficiently covered in the corpus
- Reranking: Faithfulness 0.0 — same as LLM hallucinating outside the context
- Limitation: Fixed-size chunking cuts off the context in the middle of a sentence