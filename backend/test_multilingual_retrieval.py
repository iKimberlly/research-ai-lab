
from pathlib import Path

import chromadb

from app.rag.embeddings import EmbeddingModel


CHROMA_DIR = Path("data") / "chroma_multilingual"
COLLECTION_NAME = "research_documents_multilingual"

model = EmbeddingModel()

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

questions = [
    "O que é gamificação?",
    "Quais são os efeitos da gamificação nas emoções dos estudantes?",
    "Quais métodos foram utilizados no reconhecimento de emoções?",
]

for question in questions:
    query_embedding = model.encode([question])[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=5
    )

    print("\n" + "=" * 70)
    print("PERGUNTA:", question)
    print("=" * 70)

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        print(f"\nArquivo: {metadata.get('filename')}")
        print(f"Chunk: {metadata.get('chunk_index')}")
        print(f"Distância: {distance:.4f}")
        print("Trecho:", document[:500].replace("\n", " "))
