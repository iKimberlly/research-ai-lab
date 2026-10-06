from app.llm.client import LLMClient
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore


class RAGPipeline:

    def __init__(self, top_k: int = 3):

        self.vector_store = VectorStore()

        self.retriever = Retriever(
            self.vector_store
        )

        self.llm = LLMClient()

        self.top_k = top_k

    def ask(
        self,
        question: str
    ):

        # ====================================================
        # 1. BUSCA SEMÂNTICA
        # ====================================================

        results = self.retriever.search(
            query=question,
            top_k=self.top_k
        )

        documents = results["documents"][0]

        metadatas = results["metadatas"][0]

        distances = results["distances"][0]

        # ====================================================
        # 2. CONSTRUIR CONTEXTO
        # ====================================================

        context_parts = []

        sources = []

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances
        ):

            chunk_index = metadata["chunk_index"]

            source = metadata["source"]

            context_parts.append(
                f"""
[CHUNK {chunk_index}]
{document}
"""
            )

            sources.append({
                "chunk": chunk_index,
                "source": source,
                "distance": round(
                    float(distance),
                    4
                )
            })

        context = "\n".join(
            context_parts
        )

        # ====================================================
        # 3. PROMPT DA LLM
        # ====================================================

        prompt = f"""
Você é um assistente acadêmico especializado
em análise de artigos científicos.

Responda à pergunta utilizando SOMENTE
as informações presentes no contexto fornecido.

Não invente informações.

Se a resposta não estiver presente no contexto,
diga claramente:

"A informação não foi encontrada nos documentos."

Apresente a resposta de forma:
- acadêmica;
- objetiva;
- clara;
- bem estruturada.

Pergunta:
{question}

Contexto recuperado:
{context}
"""

        # ====================================================
        # 4. GERAR RESPOSTA
        # ====================================================

        answer = self.llm.generate(
            prompt
        )

        # ====================================================
        # 5. RETORNAR RESPOSTA + FONTES
        # ====================================================

        return {
            "question": question,
            "answer": answer,
            "sources": sources
        }