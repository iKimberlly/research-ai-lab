
from app.llm.client import LLMClient
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore


GAMIFICATION_SOURCE = (
    "eebf069750484af5975b84ba734ef0ac_"
    "The effects of gamification on students emotions.pdf"
)

EMOTION_SOURCE = (
    "Reconhecimento+de+Emoções+como+ferramenta+de+"
    "apoio+às+terapias+personalizadas.pdf"
)


class RAGPipeline:
    def __init__(self, top_k: int = 3):
        self.vector_store = VectorStore(
            persist_directory="data/chroma_multilingual",
            collection_name="research_documents_multilingual",
        )

        self.retriever = Retriever(self.vector_store)
        self.llm = LLMClient()
        self.top_k = top_k

    def resolve_source(self, question: str) -> str | None:
        q = question.casefold()

        # Comparações e perguntas gerais precisam consultar
        # mais de um documento.
        broad_terms = (
            "compare",
            "comparar",
            "comparação",
            "ambos os artigos",
            "todos os documentos",
            "todos os artigos",
            "entre os artigos",
        )

        if any(term in q for term in broad_terms):
            return None

        gamification_terms = (
            "gamificação",
            "gamification",
            "gamificado",
            "gamified",
            "estudo sobre gamificação",
            "artigo de gamificação",
        )

        emotion_terms = (
            "reconhecimento de emoções",
            "reconhecimento facial",
            "terapias personalizadas",
            "fer2013",
            "vgg19",
        )

        mentions_gamification = any(
            term in q for term in gamification_terms
        )

        mentions_emotion = any(
            term in q for term in emotion_terms
        )

        # Se a pergunta indicar os dois assuntos,
        # não restringir a pesquisa a um único documento.
        if mentions_gamification and mentions_emotion:
            return None

        if mentions_gamification:
            return GAMIFICATION_SOURCE

        if mentions_emotion:
            return EMOTION_SOURCE

        return None

    def ask(self, question: str):
        source_filter = self.resolve_source(question)

        results = self.retriever.search(
            query=question,
            top_k=self.top_k,
            source=source_filter,
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        context_parts = []
        sources = []

        for document, metadata, distance in zip(
            documents, metadatas, distances
        ):
            chunk_index = metadata["chunk_index"]
            source = metadata["source"]

            context_parts.append(
                f"[CHUNK {chunk_index}]\n{document}"
            )

            sources.append(
                {
                    "chunk": chunk_index,
                    "source": metadata.get("filename", source),
                    "distance": round(float(distance), 4),
                }
            )

        context = "\n\n".join(context_parts)

        prompt = f"""
Você é um assistente acadêmico especializado
em análise de artigos científicos.

Responda à pergunta usando somente as informações
presentes no contexto fornecido.

Não invente informações.

Se a resposta não estiver presente no contexto,
diga claramente:
"A informação não foi encontrada nos documentos."

Apresente uma resposta acadêmica, objetiva,
clara e bem estruturada.

Pergunta:
{question}

Contexto recuperado:
{context}
"""

        answer = self.llm.generate(prompt)

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
        }
