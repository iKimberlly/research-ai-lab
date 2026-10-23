
from app.rag.pipeline import RAGPipeline

pipeline = RAGPipeline(top_k=5)

questions = [
    "O que é gamificação?",
    "What is gamification?",
    "What are the effects of gamification on students' emotions?",
]

for question in questions:
    print("\n" + "=" * 70)
    print(f"PERGUNTA: {question}")
    print("=" * 70)

    results = pipeline.retriever.search(
        query=question,
        top_k=5
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for document, metadata, distance in zip(
        documents, metadatas, distances
    ):
        print(f"\nChunk: {metadata.get('chunk_index')}")
        print(f"Distância: {float(distance):.4f}")
        print(f"Arquivo: {metadata.get('filename')}")
        print("Texto:")
        print(document[:500].replace("\n", " "))
        print("-" * 50)
