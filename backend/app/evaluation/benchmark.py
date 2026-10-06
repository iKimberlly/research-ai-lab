import csv
import json
from pathlib import Path

from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore

from app.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank
)


# ============================================================
# DIRETÓRIOS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]

QUESTIONS_PATH = (
    BASE_DIR
    / "data"
    / "evaluation"
    / "questions.json"
)

RESULTS_DIR = (
    BASE_DIR
    / "data"
    / "evaluation"
)

RESULTS_PATH = (
    RESULTS_DIR
    / "benchmark_results.csv"
)


# ============================================================
# CARREGAR PERGUNTAS
# ============================================================

def load_questions():

    with open(
        QUESTIONS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# SALVAR CSV
# ============================================================

def save_results_csv(results):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "id",
        "question",
        "top_k",
        "retrieved_chunks",
        "relevant_chunks",
        "distances",
        "precision_at_k",
        "recall_at_k",
        "mrr"
    ]

    with open(
        RESULTS_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in results:

            writer.writerow({
                "id": result["id"],
                "question": result["question"],
                "top_k": result["top_k"],
                "retrieved_chunks": str(
                    result["retrieved_chunks"]
                ),
                "relevant_chunks": str(
                    result["relevant_chunks"]
                ),
                "distances": str(
                    result["distances"]
                ),
                "precision_at_k":
                    result["precision_at_k"],
                "recall_at_k":
                    result["recall_at_k"],
                "mrr":
                    result["mrr"]
            })

    print(
        f"\nCSV salvo em:\n{RESULTS_PATH}"
    )


# ============================================================
# BENCHMARK
# ============================================================

def run_benchmark(top_k: int = 3):

    vector_store = VectorStore()

    print(
        f"\nDocumentos no ChromaDB: "
        f"{vector_store.collection.count()}"
    )

    retriever = Retriever(
        vector_store
    )

    questions = load_questions()

    results = []

    for item in questions:

        question = item["question"]

        print("\n" + "=" * 70)
        print(
            f"PERGUNTA: {question}"
        )
        print("=" * 70)

        search_results = retriever.search(
            query=question,
            top_k=top_k
        )

        metadatas = search_results.get(
            "metadatas",
            [[]]
        )[0]

        distances = search_results.get(
            "distances",
            [[]]
        )[0]

        retrieved_chunks = [
            metadata["chunk_index"]
            for metadata in metadatas
        ]

        relevant_chunks = set(
            item.get(
                "relevant_chunks",
                []
            )
        )

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

        print(
            f"Resultados encontrados: "
            f"{len(metadatas)}"
        )

        print(
            f"Chunks recuperados: "
            f"{retrieved_chunks}"
        )

        print(
            f"Chunks relevantes: "
            f"{sorted(relevant_chunks)}"
        )

        print(
            f"Precision@{top_k}: "
            f"{precision:.3f}"
        )

        print(
            f"Recall@{top_k}: "
            f"{recall:.3f}"
        )

        print(
            f"MRR: "
            f"{mrr:.3f}"
        )

        results.append({

            "id": item["id"],

            "question": question,

            "top_k": top_k,

            "retrieved_chunks":
                retrieved_chunks,

            "distances":
                distances,

            "relevant_chunks":
                list(relevant_chunks),

            "precision_at_k":
                precision,

            "recall_at_k":
                recall,

            "mrr":
                mrr
        })

    return results


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    results = run_benchmark(
        top_k=3
    )

    save_results_csv(
        results
    )

    print("\n")
    print("=" * 70)
    print("BENCHMARK FINALIZADO")
    print("=" * 70)

    print(
        f"\nTotal de perguntas: "
        f"{len(results)}"
    )