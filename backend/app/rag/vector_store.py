from pathlib import Path

import chromadb


# ============================================================
# RAIZ DO PROJETO
# research-ai-lab/
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]

DEFAULT_CHROMA_DIR = BASE_DIR / "data" / "chroma"


class VectorStore:

    def __init__(
        self,
        persist_directory: str | None = None
    ):

        # Se não for informado um diretório,
        # utiliza o ChromaDB padrão do projeto.
        if persist_directory is None:

            chroma_dir = DEFAULT_CHROMA_DIR

        else:

            # Converte o caminho informado em absoluto.
            chroma_dir = Path(
                persist_directory
            )

            if not chroma_dir.is_absolute():

                chroma_dir = BASE_DIR / chroma_dir

        self.chroma_dir = chroma_dir

        self.client = chromadb.PersistentClient(
            path=str(self.chroma_dir)
        )

        self.collection = (
            self.client.get_or_create_collection(
                name="research_documents"
            )
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
            query_embeddings=[
                query_embedding.tolist()
            ],
            n_results=top_k
        )