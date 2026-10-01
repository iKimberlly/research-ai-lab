from app.rag.embeddings import EmbeddingModel
from app.rag.vector_store import VectorStore


class Retriever:

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.embedding_model = EmbeddingModel()

    def search(
        self,
        query: str,
        top_k: int = 3
    ):
        """
        Busca os chunks semanticamente mais relevantes
        para uma determinada pergunta.
        """

        query_embedding = self.embedding_model.encode(
            [query]
        )[0]

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )

        return results