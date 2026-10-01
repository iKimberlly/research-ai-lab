from pathlib import Path

from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore


BASE_DIR = Path(__file__).resolve().parent.parent


# Conecta ao banco vetorial existente
vector_store = VectorStore(
    persist_directory=str(
        BASE_DIR / "data" / "chroma"
    )
)


retriever = Retriever(vector_store)


# Pergunta experimental
query = (
    "Quais métodos são utilizados para "
    "reconhecimento de emoções?"
)


print("\n===== PERGUNTA =====")
print(query)


# Busca os 3 chunks mais relevantes
results = retriever.search(
    query=query,
    top_k=3
)


print("\n===== RESULTADOS =====")


documents = results["documents"][0]
metadatas = results["metadatas"][0]
distances = results["distances"][0]


for i, (document, metadata, distance) in enumerate(
    zip(documents, metadatas, distances),
    start=1
):

    print(f"\n--- RESULTADO {i} ---")

    print(
        f"Fonte: {metadata['source']}"
    )

    print(
        f"Chunk: {metadata['chunk_index']}"
    )

    print(
        f"Distância: {distance:.4f}"
    )

    print("\nTexto:")

    print(document[:1000])