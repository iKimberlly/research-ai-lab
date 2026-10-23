
from app.rag.embeddings import EmbeddingModel
from app.rag.vector_store import VectorStore


class Retriever:
    def __init__(self, vector_store: VectorStore):
        self.embedding_model = EmbeddingModel()
        self.vector_store = vector_store

    def search(
        self,
        query: str,
        top_k: int = 3,
        source: str | None = None,
    ):
        query_embedding = self.embedding_model.encode(
            [query]
        )[0]

        return self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            source=source,
        )
