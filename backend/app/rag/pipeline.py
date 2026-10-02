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

    def ask(self, question: str):

        # ==========================================
        # 1. BUSCA SEMÂNTICA
        # ==========================================

        results = self.retriever.search(
            query=question,
            top_k=self.top_k
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        # ==========================================
        # 2. ORGANIZAR OS TRECHOS RECUPERADOS
        # ==========================================

        retrieved_chunks = []

        context_parts = []

        for i, document in enumerate(documents):

            metadata = metadatas[i]
            distance = distances[i]

            source = metadata["source"]
            chunk_index = metadata["chunk_index"]

            # Explicação baseada no mecanismo de recuperação
            reason = (
                "Trecho selecionado pela busca semântica "
                "por apresentar proximidade com a pergunta "
                "no espaço de embeddings."
            )

            retrieved_chunks.append({
                "rank": i + 1,
                "source": source,
                "chunk_index": chunk_index,
                "distance": round(distance, 4),
                "content": document,
                "reason": reason
            })

            context_parts.append(
                f"""
[Fonte: {source}
Chunk: {chunk_index}
Distância: {distance:.4f}]

{document}
"""
            )

        context = "\n".join(context_parts)

        # ==========================================
        # 3. PROMPT RAG
        # ==========================================

        prompt = f"""
Você é um assistente acadêmico especializado
em análise de artigos científicos.

Responda à pergunta utilizando SOMENTE as
informações presentes no contexto fornecido.

Não invente informações.

Se a resposta não puder ser encontrada no
contexto, diga claramente:

"A informação não foi encontrada nos documentos."

Apresente a resposta de forma acadêmica,
objetiva e clara.

Não utilize conhecimento externo ao contexto.

====================
CONTEXTO RECUPERADO
====================

{context}

====================
PERGUNTA
====================

{question}
"""

        # ==========================================
        # 4. GERAÇÃO DA RESPOSTA
        # ==========================================

        answer = self.llm.generate(prompt)

        # ==========================================
        # 5. RETORNO COMPLETO
        # ==========================================

        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks
        }