
from app.llm.client import LLMClient
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore


class RAGPipeline:
    def __init__(self, top_k: int = 5):
        self.vector_store = VectorStore()
        self.retriever = Retriever(self.vector_store)
        self.llm = LLMClient()
        self.top_k = top_k

    def ask(self, question: str):
        # 1. Recuperar trechos relevantes
        results = self.retriever.search(
            query=question,
            top_k=self.top_k
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        # 2. Construir contexto e fontes
        context_parts = []
        sources = []

        for document, metadata, distance in zip(
            documents, metadatas, distances
        ):
            if not document or not document.strip():
                continue

            chunk_index = metadata.get("chunk_index")
            source = metadata.get("source", "Documento desconhecido")

            context_parts.append(
                f"[CHUNK {chunk_index}]\n{document}"
            )

            sources.append({
                "chunk": chunk_index,
                "source": source,
                "distance": round(float(distance), 4)
            })

        context = "\n\n".join(context_parts)

        # 3. Não gerar resposta sem contexto
        if not context.strip():
            return {
                "question": question,
                "answer": (
                    "Não foi possível encontrar trechos relevantes "
                    "nos documentos indexados para responder à pergunta."
                ),
                "sources": []
            }

        # 4. Instruções de fundamentação
        system_prompt = """
Você é um assistente especializado em análise de artigos científicos.

Regras obrigatórias:
- Responda em português claro e acadêmico.
- Utilize somente informações sustentadas pelo contexto fornecido.
- Não invente métodos, resultados, autores, números ou conclusões.
- Não apresente conhecimento geral como se estivesse no artigo.
- Se a informação solicitada não estiver no contexto, diga:
  "A informação não foi encontrada nos trechos recuperados."
- Diferencie fatos explícitos de inferências.
- Não afirme que o artigo não contém uma informação apenas porque
  ela não aparece nos trechos recuperados.
- Organize a resposta em tópicos quando isso facilitar a leitura.
"""

        user_prompt = f"""
Pergunta:
{question}

Contexto recuperado:
<contexto>
{context}
</contexto>

Responda com base exclusivamente no contexto.
"""

        # 5. Gerar resposta
        answer = self.llm.generate(
            user_prompt,
            system_prompt=system_prompt
        )

        # 6. Retornar resposta e fontes
        return {
            "question": question,
            "answer": answer.strip(),
            "sources": sources
        }
