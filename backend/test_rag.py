from app.rag.pipeline import RAGPipeline


pipeline = RAGPipeline(top_k=3)


question = (
    "Quais métodos são utilizados para "
    "reconhecimento de emoções?"
)


result = pipeline.ask(question)


# ==========================================
# PERGUNTA
# ==========================================

print("\n" + "=" * 70)
print("PERGUNTA")
print("=" * 70)

print(result["question"])


# ==========================================
# TRECHOS RECUPERADOS
# ==========================================

print("\n" + "=" * 70)
print("TRECHOS RECUPERADOS PELO RAG")
print("=" * 70)


for chunk in result["retrieved_chunks"]:

    print("\n" + "-" * 70)

    print(
        f"RANKING: #{chunk['rank']}"
    )

    print(
        f"DISTÂNCIA: {chunk['distance']}"
    )

    print(
        f"CHUNK: {chunk['chunk_index']}"
    )

    print(
        f"FONTE: {chunk['source']}"
    )

    print("\nPOR QUE FOI RECUPERADO:")

    print(chunk["reason"])

    print("\nTRECHO DO ARTIGO:")

    print(chunk["content"])


# ==========================================
# RESPOSTA DA LLM
# ==========================================

print("\n" + "=" * 70)
print("RESPOSTA DA LLM")
print("=" * 70)

print(result["answer"])