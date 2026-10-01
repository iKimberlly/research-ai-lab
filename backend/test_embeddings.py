from app.rag.loader import load_pdf
from app.rag.chunker import chunk_text
from app.rag.embeddings import EmbeddingModel

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = BASE_DIR / "data" / "documents"

pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError(
        "Nenhum PDF encontrado."
    )

pdf_path = pdf_files[0]


# 1. Extrair texto
text = load_pdf(str(pdf_path))


# 2. Criar chunks
chunks = chunk_text(
    text,
    chunk_size=1000,
    overlap=200
)


print(f"Chunks encontrados: {len(chunks)}")


# 3. Criar modelo
embedding_model = EmbeddingModel()


# 4. Gerar embeddings
embeddings = embedding_model.encode(
    chunks[:3]
)


print("\n===== EMBEDDINGS =====")

for i, embedding in enumerate(embeddings, start=1):

    print(f"\nChunk {i}")
    print(f"Dimensão: {len(embedding)}")
    print(embedding[:10])