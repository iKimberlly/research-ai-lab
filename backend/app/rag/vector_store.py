
import chromadb


class VectorStore:

    def __init__(self, persist_directory="data/chroma"):
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
        source: str,
        filename: str = ""
    ):
        ids = [
            f"{source}_{i}"
            for i in range(len(chunks))
        ]

        metadatas = [
            {
                "source": source,
                "filename": filename or source,
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
        top_k: int = 3,
        source: str | None = None
    ):
        query_args = {
            "query_embeddings": [
                query_embedding.tolist()
            ],
            "n_results": min(
                top_k,
                self.collection.count()
            )
        }

        if source:
            query_args["where"] = {
                "source": source
            }

        if query_args["n_results"] == 0:
            return {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
                "ids": [[]]
            }

        return self.collection.query(**query_args)
