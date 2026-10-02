import json
from pathlib import Path

from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore

from app.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank
)


BASE_DIR = Path(__file__).resolve().parents[3]

QUESTIONS_PATH = (
    BASE_DIR
    / "data"
    / "evaluation"
    / "questions.json"
)


def load_questions():

    with open(
        QUESTIONS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def run_benchmark(top_k: int = 3):

    vector_store = VectorStore()

    retriever = Retriever(
        vector_store
    )

    questions = load_questions()

    results = []

    for item in questions:

        question = item["question"]

        relevant_chunks = set(
            item["relevant_chunks"]
        )

        search_results = retriever.search(
            query=question,
            top_k=top_k
        )

        retrieved_chunks = [
            metadata["chunk_index"]
            for metadata in search_results["metadatas"][0]
        ]

        precision = precision_at_k(
            retrieved_chunks,
            relevant_chunks,
            top_k
        )

        recall = recall_at_k(
            retrieved_chunks,
            relevant_chunks,
            top_k
        )

        mrr = reciprocal_rank(
            retrieved_chunks,
            relevant_chunks
        )

        results.append({
            "id": item["id"],
            "question": question,
            "retrieved_chunks": retrieved_chunks,
            "relevant_chunks": list(
                relevant_chunks
            ),
            "precision_at_k": precision,
            "recall_at_k": recall,
            "mrr": mrr
        })

    return results


if __name__ == "__main__":

    results = run_benchmark(
        top_k=3
    )

    print("\n" + "=" * 70)
    print("BENCHMARK DO RETRIEVER")
    print("=" * 70)

    for result in results:

        print("\n" + "-" * 70)

        print(
            f"Pergunta: {result['question']}"
        )

        print(
            f"Chunks recuperados: "
            f"{result['retrieved_chunks']}"
        )

        print(
            f"Chunks relevantes: "
            f"{result['relevant_chunks']}"
        )

        print(
            f"Precision@3: "
            f"{result['precision_at_k']:.3f}"
        )

        print(
            f"Recall@3: "
            f"{result['recall_at_k']:.3f}"
        )

        print(
            f"MRR: "
            f"{result['mrr']:.3f}"
        )