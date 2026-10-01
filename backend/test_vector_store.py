from pathlib import Path

from app.rag.loader import load_pdf
from app.rag.chunker import chunk_text
from app.rag.embeddings import EmbeddingModel
from app.rag.vector_store import VectorStore


BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = BASE_DIR / "data" / "documents"

pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError(
        "Nenhum PDF encontrado."
    )

pdf_path = pdf_files[0]


# ==========================================
# 1. EXTRAÇÃO
# ==========================================

text = load_pdf(str(pdf_path))


# ==========================================
# 2. CHUNKING
# ==========================================

chunks = chunk_text(
    text,
    chunk_size=1000,
    overlap=200
)

print(f"Chunks encontrados: {len(chunks)}")


# ==========================================
# 3. EMBEDDINGS
# ==========================================

embedding_model = EmbeddingModel()

embeddings = embedding_model.encode(chunks)

print(
    f"Embeddings gerados: {len(embeddings)}"
)

print(
    f"Dimensão dos embeddings: {len(embeddings[0])}"
)


# ==========================================
# 4. CHROMADB
# ==========================================

vector_store = VectorStore(
    persist_directory=str(
        BASE_DIR / "data" / "chroma"
    )
)


vector_store.add_documents(
    chunks=chunks,
    embeddings=embeddings,
    source=pdf_path.name
)


print("\n===== CHROMADB =====")
print("Documentos armazenados com sucesso.")


# ==========================================
# 5. VERIFICAR
# ==========================================

count = vector_store.collection.count()

print(f"Total de documentos no banco: {count}")