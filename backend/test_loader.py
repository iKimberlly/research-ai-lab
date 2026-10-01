from pathlib import Path

from app.rag.loader import load_pdf
from app.rag.chunker import chunk_text


BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = BASE_DIR / "data" / "documents"

pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError(
        f"Nenhum PDF encontrado em: {DOCUMENTS_DIR}"
    )

pdf_path = pdf_files[0]

print(f"\nPDF encontrado: {pdf_path.name}")

# 1. Extrair texto
text = load_pdf(str(pdf_path))

print("\n===== DOCUMENTO =====")
print(f"Caracteres: {len(text)}")
print(f"Palavras aproximadas: {len(text.split())}")


# 2. Dividir em chunks
chunks = chunk_text(
    text,
    chunk_size=1000,
    overlap=200
)

print("\n===== CHUNKING =====")
print(f"Total de chunks: {len(chunks)}")


# 3. Mostrar alguns exemplos
for i, chunk in enumerate(chunks[:3], start=1):

    print(f"\n===== CHUNK {i} =====")
    print(f"Tamanho: {len(chunk)} caracteres")
    print(chunk[:1000])