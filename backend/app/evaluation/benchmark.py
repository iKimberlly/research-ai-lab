
import csv
import json
from pathlib import Path

from app.rag.embeddings import EmbeddingModel
from app.rag.vector_store import VectorStore


# ============================================================
# CAMINHOS
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parents[2]

EVALUATION_DIR = BACKEND_DIR / "data" / "evaluation"

QUESTIONS_PATH = EVALUATION_DIR / "questions.json"
RESULTS_PATH = EVALUATION_DIR / "benchmark_results.csv"
SUMMARY_PATH = EVALUATION_DIR / "benchmark_summary.csv"


# ============================================================
# MAPEAMENTO DAS FONTES
# Os nomes devem corresponder ao metadata "filename"
# armazenado no ChromaDB.
# ============================================================

SOURCE_FILENAMES = {
    "gamification": (
        "eebf069750484af5975b84ba734ef0ac_The effects of gamifcation os student_s emotions.pdf"
    ),
    "emotion_recognition": (
        "Reconhecimento+de+Emoções+como+ferramenta+de+apoio+às+terapias+personalizadas.pdf"
    ),
}

# ============================================================
# CARREGAMENTO
# ============================================================

def load_questions():
    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo de perguntas não encontrado: {QUESTIONS_PATH}"
        )

    with QUESTIONS_PATH.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    if not isinstance(questions, list):
        raise ValueError("questions.json deve conter uma lista.")

    return questions


def source_alias(filename):
    for alias, expected_filename in SOURCE_FILENAMES.items():
        if filename == expected_filename:
            return alias

    return filename


def chunk_key(filename, chunk_index):
    """Identifica um chunk pelo documento e pelo índice."""
    return (filename, int(chunk_index))


# ============================================================
# BUSCA
# ============================================================

def search_documents(
    question,
    vector_store,
    embedding_model,
    top_k=3,
    source="both",
    use_source_filter=True,
):
    query_embedding = embedding_model.encode([question])[0]

    query_args = {
        "query_embeddings": [query_embedding.tolist()],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }

    if use_source_filter and source != "both":
        if source not in SOURCE_FILENAMES:
            raise ValueError(f"Fonte desconhecida: {source}")

        query_args["where"] = {
            "filename": SOURCE_FILENAMES[source]
        }

    results = vector_store.collection.query(**query_args)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved = []

    for document, metadata, distance in zip(
        documents, metadatas, distances
    ):
        filename = metadata["filename"]
        chunk_index = int(metadata["chunk_index"])

        retrieved.append({
            "filename": filename,
            "source": source_alias(filename),
            "chunk_index": chunk_index,
            "key": chunk_key(filename, chunk_index),
            "distance": float(distance),
            "text": document,
        })

    return retrieved


# ============================================================
# MÉTRICAS
# ============================================================

def calculate_metrics(retrieved, relevant_chunks, k):
    """
    relevant_chunks deve conter objetos:
    {"source": "gamification", "chunk_index": 5}
    """

    relevant_keys = set()

    for item in relevant_chunks:
        if isinstance(item, dict):
            source = item.get("source")
            index = item.get("chunk_index")

            if source not in SOURCE_FILENAMES:
                continue

            relevant_keys.add(
                chunk_key(SOURCE_FILENAMES[source], index)
            )

    retrieved_keys = [item["key"] for item in retrieved[:k]]

    # Não calcula métricas de relevância sem ground truth.
    if not relevant_keys:
        return {
            "precision_at_k": None,
            "recall_at_k": None,
            "mrr": None,
        }

    relevant_retrieved = sum(
        key in relevant_keys for key in retrieved_keys
    )

    precision = relevant_retrieved / k if k else 0.0
    recall = relevant_retrieved / len(relevant_keys)

    mrr = 0.0

    for rank, key in enumerate(retrieved_keys, start=1):
        if key in relevant_keys:
            mrr = 1 / rank
            break

    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "mrr": mrr,
    }


# ============================================================
# BENCHMARK
# ============================================================

def run_benchmark(top_k=3, use_source_filter=True):
    vector_store = VectorStore(
        persist_directory="data/chroma_multilingual",
        collection_name="research_documents_multilingual",
    )
    embedding_model = EmbeddingModel()

    print(f"Chunks na coleção: {vector_store.collection.count()}")
    print(f"Filtro de fontes: {'ATIVADO' if use_source_filter else 'DESATIVADO'}")
    print(f"Top K: {top_k}")

    questions = load_questions()
    results = []

    for item in questions:
        question = item["question"]
        source = item.get("source", "both")

        print("\n" + "=" * 70)
        print(f"Pergunta: {question}")
        print(f"Fonte esperada: {source}")

        retrieved = search_documents(
            question=question,
            vector_store=vector_store,
            embedding_model=embedding_model,
            top_k=top_k,
            source=source,
            use_source_filter=use_source_filter,
        )

        metrics = calculate_metrics(
            retrieved=retrieved,
            relevant_chunks=item.get("relevant_chunks", []),
            k=top_k,
        )

        for rank, result in enumerate(retrieved, start=1):
            print(
                f"{rank}. {result['filename']} | "
                f"chunk={result['chunk_index']} | "
                f"distância={result['distance']:.4f}"
            )

        print(f"Precision@{top_k}: {metrics['precision_at_k']}")
        print(f"Recall@{top_k}: {metrics['recall_at_k']}")
        print(f"MRR: {metrics['mrr']}")

        results.append({
            "id": item["id"],
            "question": question,
            "expected_source": source,
            "filter_enabled": use_source_filter,
            "top_k": top_k,
            "retrieved_chunks": json.dumps(
                [
                    {
                        "source": r["source"],
                        "filename": r["filename"],
                        "chunk_index": r["chunk_index"],
                    }
                    for r in retrieved
                ],
                ensure_ascii=False,
            ),
            "distances": json.dumps(
                [round(r["distance"], 4) for r in retrieved]
            ),
            "relevant_chunks": json.dumps(
                item.get("relevant_chunks", []),
                ensure_ascii=False,
            ),
            **metrics,
        })

    return results


# ============================================================
# SALVAR RESULTADOS
# ============================================================

def save_results(results, path=RESULTS_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not results:
        print("Nenhum resultado para salvar.")
        return

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResultados salvos em: {path}")


def save_summary(results):
    metric_names = [
        "precision_at_k",
        "recall_at_k",
        "mrr",
    ]

    summary = []

    for filter_enabled in (True, False):
        group = [
            row for row in results
            if row["filter_enabled"] == filter_enabled
        ]

        for metric in metric_names:
            values = [
                row[metric] for row in group
                if row[metric] is not None
            ]

            summary.append({
                "filter_enabled": filter_enabled,
                "metric": metric,
                "mean": (
                    sum(values) / len(values)
                    if values else None
                ),
                "questions_evaluated": len(values),
                "questions_without_ground_truth": (
                    len(group) - len(values)
                ),
            })

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    with SUMMARY_PATH.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "filter_enabled",
                "metric",
                "mean",
                "questions_evaluated",
                "questions_without_ground_truth",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)

    print(f"Resumo salvo em: {SUMMARY_PATH}")


if __name__ == "__main__":
    filtered_results = run_benchmark(
        top_k=3,
        use_source_filter=True,
    )

    unfiltered_results = run_benchmark(
        top_k=3,
        use_source_filter=False,
    )

    all_results = filtered_results + unfiltered_results

    save_results(all_results)
    save_summary(all_results)

    print("\nBenchmark concluído.")
