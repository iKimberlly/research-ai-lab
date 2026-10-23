
from pathlib import Path
import chromadb

from app.rag.loader import load_pdf
from app.rag.chunker import chunk_text
from app.rag.embeddings import EmbeddingModel


# Caminhos relativos à pasta backend/
DOCUMENTS_DIR = Path("..") / "data" / "documents"
CHROMA_DIR = Path("data") / "chroma_multilingual"

COLLECTION_NAME = "research_documents_multilingual"


def main():
    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(
            f"Pasta de documentos não encontrada: {DOCUMENTS_DIR.resolve()}"
        )

    pdf_files = sorted(DOCUMENTS_DIR.rglob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"Nenhum PDF encontrado em {DOCUMENTS_DIR.resolve()}"
        )

    print(f"PDFs encontrados: {len(pdf_files)}")
    print(f"Base nova: {CHROMA_DIR.resolve()}")

    embedding_model = EmbeddingModel()

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    for pdf_path in pdf_files:
        print(f"\nProcessando: {pdf_path.name}")

        try:
            text = load_pdf(str(pdf_path))

            if len(text.strip()) < 50:
                print("Ignorado: texto extraído insuficiente.")
                continue

            chunks = chunk_text(
                text,
                chunk_size=1000,
                overlap=200
            )

            if not chunks:
                print("Ignorado: nenhum chunk gerado.")
                continue

            embeddings = embedding_model.encode(chunks)

            # Identificador estável para este arquivo nesta coleção.
            source_id = pdf_path.name

            ids = [
                f"{source_id}_{i}"
                for i in range(len(chunks))
            ]

            metadatas = [
                {
                    "source": source_id,
                    "filename": pdf_path.name,
                    "chunk_index": i
                }
                for i in range(len(chunks))
            ]

            collection.upsert(
                ids=ids,
                documents=chunks,
                embeddings=embeddings.tolist(),
                metadatas=metadatas
            )

            print(f"Caracteres extraídos: {len(text)}")
            print(f"Chunks indexados: {len(chunks)}")

        except Exception as error:
            print(f"ERRO em {pdf_path.name}: {error}")

    print("\n===== RESUMO =====")
    print(f"Coleção: {COLLECTION_NAME}")
    print(f"Total de chunks: {collection.count()}")
    print(f"Banco: {CHROMA_DIR.resolve()}")


if __name__ == "__main__":
    main()
