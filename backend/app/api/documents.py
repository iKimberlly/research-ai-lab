
from pathlib import Path
from uuid import uuid4
import re
import shutil

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.rag.loader import load_pdf
from app.rag.chunker import chunk_text
from app.rag.embeddings import EmbeddingModel
from app.rag.vector_store import VectorStore


router = APIRouter(
    prefix="/api/documents",
    tags=["Documents"]
)

BASE_DIR = Path(__file__).resolve().parents[3]
UPLOAD_DIR = BASE_DIR / "data" / "documents"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

embedding_model = EmbeddingModel()


def safe_filename(filename: str) -> str:
    """Remove caracteres problemáticos do nome do arquivo."""
    name = Path(filename).name
    name = re.sub(r"[^\w.\- ]", "_", name)
    return name[:150] or "artigo.pdf"


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Envie um arquivo PDF."
        )

    document_id = uuid4().hex
    filename = safe_filename(file.filename)

    # O ID exclusivo evita colisões entre arquivos de mesmo nome.
    stored_path = UPLOAD_DIR / f"{document_id}_{filename}"

    try:
        total_size = 0

        with stored_path.open("wb") as destination:
            while True:
                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="O arquivo excede o limite de 20 MB."
                    )

                destination.write(chunk)

        if total_size == 0:
            raise HTTPException(
                status_code=400,
                detail="O arquivo enviado está vazio."
            )

        # 1. Extrair texto do PDF
        text = load_pdf(str(stored_path))

        if len(text.strip()) < 50:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Não foi possível extrair texto suficiente. "
                    "Se o PDF for digitalizado, será necessário adicionar OCR."
                )
            )

        # 2. Dividir o texto em chunks
        chunks = chunk_text(
            text,
            chunk_size=1000,
            overlap=200
        )

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="Nenhum trecho foi gerado a partir do PDF."
            )

        # 3. Gerar embeddings
        embeddings = embedding_model.encode(chunks)

        # 4. Armazenar no ChromaDB
        vector_store = VectorStore()

        vector_store.add_documents(
            chunks=chunks,
            embeddings=embeddings,
            source=document_id,
            filename=filename
        )

        return {
            "document_id": document_id,
            "filename": filename,
            "characters_extracted": len(text),
            "chunks_indexed": len(chunks),
            "status": "indexed"
        }

    except HTTPException:
        stored_path.unlink(missing_ok=True)
        raise

    except Exception as exc:
        stored_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail=f"Falha ao processar o PDF: {str(exc)}"
        ) from exc

    finally:
        await file.close()
