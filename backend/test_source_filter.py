
from app.rag.pipeline import RAGPipeline

pipeline = RAGPipeline()

questions = [
    "Quais emoções foram analisadas no estudo sobre gamificação?",
    "Quais métodos foram usados no reconhecimento de emoções?",
    "Compare os dois artigos.",
    "Qual é a conclusão dos documentos?",
]

for question in questions:
    source = pipeline.resolve_source(question)

    print(f"\nPergunta: {question}")
    print(f"Filtro: {source or 'Todos os documentos'}")
