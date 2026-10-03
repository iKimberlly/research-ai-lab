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
        print(f"PERGUNTA: {question}")
        print("=" * 70)

        search_results = retriever.search(
            query=question,
            top_k=top_k
        )

        # ------------------------------------------
        # Verificar resultados da busca
        # ------------------------------------------

        metadatas = search_results.get(
            "metadatas",
            [[]]
        )[0]

        distances = search_results.get(
            "distances",
            [[]]
        )[0]

        print(
            f"Resultados encontrados: "
            f"{len(metadatas)}"
        )

        # ------------------------------------------
        # Extrair chunks
        # ------------------------------------------

        retrieved_chunks = [
            metadata["chunk_index"]
            for metadata in metadatas
        ]

        print(
            f"Chunks recuperados: "
            f"{retrieved_chunks}"
        )

        # ------------------------------------------
        # Ground truth
        # ------------------------------------------

        relevant_chunks = set(
            item.get(
                "relevant_chunks",
                []
            )
        )

        print(
            f"Chunks relevantes: "
            f"{sorted(relevant_chunks)}"
        )

        # ------------------------------------------
        # Métricas
        # ------------------------------------------

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

        # ------------------------------------------
        # Salvar resultado
        # ------------------------------------------

        results.append({

            "id": item["id"],

            "question": question,

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


if __name__ == "__main__":

    results = run_benchmark(
        top_k=3
    )

    print("\n")
    print("=" * 70)
    print("BENCHMARK FINALIZADO")
    print("=" * 70)

    print(
        f"\nTotal de perguntas: "
        f"{len(results)}"
    )