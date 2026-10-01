import chromadb


class VectorStore:

    def __init__(self, persist_directory: str = "data/chroma"):
        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name="research_documents"
        )

    def add_documents(
        self,
        chunks: list[str],
        embeddings,
        source: str
    ):
        ids = [
            f"{source}_{i}"
            for i in range(len(chunks))
        ]

        metadatas = [
            {
                "source": source,
                "chunk_index": i
            }
            for i in range(len(chunks))
        ]

        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )

    def search(
        self,
        query_embedding,
        top_k: int = 3
    ):
        return self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )