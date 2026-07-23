import os
import sys
import csv
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from groq import Groq as GroqClient
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from src.pipeline_advanced import load_advanced_index, query_advanced

load_dotenv()

GROQ_CLIENT = GroqClient(api_key=os.getenv("GROQ_API_KEY"))
EMBED_MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5")
LLM_MODEL   = "llama-3.3-70b-versatile"

# Eval dataset — SAME as Naive RAG
# MUST use same questions for valid comparison
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
        "question": "How does reranking improve RAG performance?",
        "ground_truth": "Reranking adds a second-pass model that re-evaluates the initially retrieved documents and reorders them by true relevance, reducing noise before passing context to the LLM."
    },
    {
        "question": "What is the difference between RAG and fine-tuning?",
        "ground_truth": "RAG retrieves external knowledge at inference time without modifying model weights, while fine-tuning updates model parameters on domain-specific data. RAG is more flexible for dynamic knowledge while fine-tuning is better for static domain adaptation."
    },
]


# Metrics — same as Naive RAG
# Must use identical functions for valid comparison

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


def compute_faithfulness(answer: str, contexts: list[str]) -> float:
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
        raw   = llm_call(prompt)
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
        orig_emb  = embed(question)
        gen_embs  = embed(questions)
        sims      = cosine_similarity(orig_emb, gen_embs)[0]
        return round(float(np.mean(sims)), 4)
    except Exception as e:
        print(f"  [answer_relevancy error] {e}")
        return 0.0


def compute_context_precision(question: str, contexts: list[str], ground_truth: str) -> float:
    gt_emb   = embed(ground_truth)
    ctx_embs = embed(contexts)
    sims     = cosine_similarity(gt_emb, ctx_embs)[0]
    relevant = sum(1 for s in sims if s >= 0.5)
    return round(relevant / len(contexts), 4)


def compute_context_recall(answer: str, ground_truth: str) -> float:
    ans_emb = embed(answer)
    gt_emb  = embed(ground_truth)
    sim     = cosine_similarity(ans_emb, gt_emb)[0][0]
    return round(float(sim), 4)


# ── Main Evaluation ───────────────────────────────────────
def run_evaluation():
    print("Loading Advanced RAG pipeline...")
    index, embed_model_llamaindex = load_advanced_index()

    results = []

    print("\nRunning evaluation...\n")
    for item in EVAL_DATASET:
        q  = item["question"]
        gt = item["ground_truth"]
        print(f"Q: {q[:60]}...")

        # Call the Advanced RAG pipeline
        result   = query_advanced(q, index=index, embed_model=embed_model_llamaindex)
        answer   = result["answer"]
        contexts = [s["text"] for s in result["sources"]]

        # Calculate metrics — same function as Naive RAG
        faith  = compute_faithfulness(answer, contexts)
        rel    = compute_answer_relevancy(q, answer)
        prec   = compute_context_precision(q, contexts, gt)
        recall = compute_context_recall(answer, gt)

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

    # Save to CSV
    os.makedirs("evaluation", exist_ok=True)
    with open("evaluation/advanced_rag_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Calculate averages
    avg_faith  = np.mean([r["faithfulness"]      for r in results])
    avg_rel    = np.mean([r["answer_relevancy"]   for r in results])
    avg_prec   = np.mean([r["context_precision"]  for r in results])
    avg_recall = np.mean([r["context_recall"]     for r in results])

    # Comparison Table
    # Naive RAG scores are taken from previous evaluation results.
    # Hardcoded here to allow direct comparison without opening other files.
    naive_scores = {
        "faithfulness":      0.5000,
        "answer_relevancy":  0.8549,
        "context_precision": 0.9600,
        "context_recall":    0.8503,
    }

    print("\n" + "=" * 60)
    print("COMPARISON: NAIVE RAG vs ADVANCED RAG")
    print("=" * 60)
    print(f"{'Metric':<22} {'Naive':>8} {'Advanced':>10} {'Delta':>8}")
    print("-" * 60)

    metrics = [
        ("Faithfulness",      naive_scores["faithfulness"],      avg_faith),
        ("Answer Relevancy",  naive_scores["answer_relevancy"],  avg_rel),
        ("Context Precision", naive_scores["context_precision"], avg_prec),
        ("Context Recall",    naive_scores["context_recall"],    avg_recall),
    ]

    for name, naive_val, adv_val in metrics:
        delta = adv_val - naive_val
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        print(f"{name:<22} {naive_val:>8.4f} {adv_val:>10.4f} {arrow} {abs(delta):>5.4f}")

    print("=" * 60)
    print(f"\nDetail saved to evaluation/advanced_rag_results.csv")
    print("\nThese delta values are what you should put in your CV and README!")


if __name__ == "__main__":
    run_evaluation()