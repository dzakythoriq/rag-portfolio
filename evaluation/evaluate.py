import os
import sys
import json
import csv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from groq import Groq as GroqClient
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from src.pipeline import load_index, build_query_engine, query_with_sources

load_dotenv()

GROQ_CLIENT  = GroqClient(api_key=os.getenv("GROQ_API_KEY"))
EMBED_MODEL  = SentenceTransformer("BAAI/bge-small-en-v1.5")
LLM_MODEL    = "llama-3.3-70b-versatile"

EVAL_DATASET = [
    {
        "question": "What is Retrieval-Augmented Generation?",
        "ground_truth": "Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with language model generation, where relevant documents are retrieved and provided as context to the LLM before generating an answer."
    },
    {
        "question": "What are the main limitations of Naive RAG?",
        "ground_truth": "Naive RAG suffers from retrieval noise where irrelevant chunks are retrieved, context window limitations, and semantic gap between user queries and document text despite having the same meaning."
    },
    {
        "question": "What is the difference between sparse and dense retrieval?",
        "ground_truth": "Sparse retrieval uses keyword-based methods like BM25 that rely on exact word matches, while dense retrieval uses neural embeddings to capture semantic similarity between queries and documents."
    },
    {
        "question": "What is HyDE in the context of RAG?",
        "ground_truth": "HyDE (Hypothetical Document Embeddings) is a technique where the LLM first generates a hypothetical answer to the query, which is then used as the search query in the vector store instead of the original question."
    },
    {
        "question": "How does reranking improve RAG performance?",
        "ground_truth": "Reranking adds a second-pass model that re-evaluates the initially retrieved documents and reorders them by true relevance, reducing noise before passing context to the LLM."
    },
]


def llm_call(prompt: str) -> str:
    response = GROQ_CLIENT.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0
    )
    return response.choices[0].message.content.strip()


def embed(texts):
    if isinstance(texts, str):
        texts = [texts]
    return EMBED_MODEL.encode(texts, normalize_embeddings=True)


# ── Metrik ───────────────────────────────────────────────

def compute_faithfulness(answer: str, contexts: list[str]) -> float:
    """
    Ask LLM: is every claim in the answer supported by the context?
    Score = number of supported claims / total claims
    """
    context_str = "\n\n".join(contexts)
    prompt = f"""Given the following context and answer, extract all factual claims from the answer.
Then for each claim, determine if it is supported by the context (yes/no).
Return a JSON object with this exact format:
{{"claims": [{{"claim": "...", "supported": true/false}}]}}

Context:
{context_str}

Answer:
{answer}

Return only the JSON, nothing else."""

    try:
        raw  = llm_call(prompt)
        #ambil JSON dari response
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        data  = json.loads(raw[start:end])
        claims = data.get("claims", [])
        if not claims:
            return 0.0
        supported = sum(1 for c in claims if c.get("supported", False))
        return round(supported / len(claims), 4)
    except Exception as e:
        print(f"  [faithfulness error] {e}")
        return 0.0


def compute_answer_relevancy(question: str, answer: str) -> float:
    """
    Generate several questions from the answer, then measure cosine similarity
    between those questions and the original question.
    """
    prompt = f"""Generate 3 questions that the following answer is trying to answer.
Return only a JSON array of strings.
Example: ["question 1", "question 2", "question 3"]

Answer:
{answer}

Return only the JSON array, nothing else."""

    try:
        raw       = llm_call(prompt)
        start     = raw.find("[")
        end       = raw.rfind("]") + 1
        questions = json.loads(raw[start:end])

        orig_emb = embed(question)
        gen_embs = embed(questions)
        sims     = cosine_similarity(orig_emb, gen_embs)[0]
        return round(float(np.mean(sims)), 4)
    except Exception as e:
        print(f"  [answer_relevancy error] {e}")
        return 0.0


def compute_context_precision(question: str, contexts: list[str], ground_truth: str) -> float:
    """
    Of all the retrieved chunks, what percentage is truly relevant?
    Measure with cosine similarity between the chunk and ground truth.
    """
    gt_emb  = embed(ground_truth)
    ctx_embs = embed(contexts)
    sims    = cosine_similarity(gt_emb, ctx_embs)[0]
    # threshold 0.5 = relevan
    relevant = sum(1 for s in sims if s >= 0.5)
    return round(relevant / len(contexts), 4)


def compute_context_recall(answer: str, ground_truth: str) -> float:
    """
    How much information from the ground truth is included in the answer?
    Measure with cosine similarity between the answer and ground truth.
    """
    ans_emb = embed(answer)
    gt_emb  = embed(ground_truth)
    sim     = cosine_similarity(ans_emb, gt_emb)[0][0]
    return round(float(sim), 4)


def run_evaluation():
    print("Loading RAG pipeline...")
    index        = load_index()
    query_engine = build_query_engine(index)

    results = []

    print("\nRunning evaluation...\n")
    for item in EVAL_DATASET:
        q  = item["question"]
        gt = item["ground_truth"]
        print(f"Q: {q[:60]}...")

        result   = query_with_sources(query_engine, q)
        answer   = result["answer"]
        contexts = [s["text"] for s in result["sources"]]

        faith   = compute_faithfulness(answer, contexts)
        rel     = compute_answer_relevancy(q, answer)
        prec    = compute_context_precision(q, contexts, gt)
        recall  = compute_context_recall(answer, gt)

        print(f"  Faithfulness:      {faith}")
        print(f"  Answer Relevancy:  {rel}")
        print(f"  Context Precision: {prec}")
        print(f"  Context Recall:    {recall}")
        print()

        results.append({
            "question":          q,
            "answer":            answer,
            "ground_truth":      gt,
            "faithfulness":      faith,
            "answer_relevancy":  rel,
            "context_precision": prec,
            "context_recall":    recall,
        })

    # Simpan CSV
    os.makedirs("evaluation", exist_ok=True)
    with open("evaluation/naive_rag_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Summary
    avg_faith  = np.mean([r["faithfulness"]      for r in results])
    avg_rel    = np.mean([r["answer_relevancy"]   for r in results])
    avg_prec   = np.mean([r["context_precision"]  for r in results])
    avg_recall = np.mean([r["context_recall"]     for r in results])

    print("=" * 50)
    print("NAIVE RAG BASELINE SCORES")
    print("=" * 50)
    print(f"Faithfulness:      {avg_faith:.4f}")
    print(f"Answer Relevancy:  {avg_rel:.4f}")
    print(f"Context Precision: {avg_prec:.4f}")
    print(f"Context Recall:    {avg_recall:.4f}")
    print("=" * 50)
    print("\nThe details are stored in evaluation/naive_rag_results.csv")
    print("KEEP THESE NUMBERS — baseline for Advanced RAG later!")


if __name__ == "__main__":
    run_evaluation()