
from pathlib import Path
import io

import fitz
import pytesseract
from PIL import Image


# Caminho do executável instalado no Windows.
TESSERACT_PATH = Path(
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

if TESSERACT_PATH.exists():
    pytesseract.pytesseract.tesseract_cmd = str(
        TESSERACT_PATH
    )


def extract_page_text(page) -> str:
    """
    Extrai o texto de uma página.
    Se a página não tiver texto suficiente, utiliza OCR.
    """

    text = page.get_text("text").strip()

    # PDFs digitais: aproveita o texto já existente.
    if len(text) >= 30:
        return text

    # PDFs digitalizados: converte a página em imagem.
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(2.5, 2.5),
        alpha=False
    )

    image_bytes = pixmap.tobytes("png")
    image = Image.open(io.BytesIO(image_bytes))

    # Reconhece texto em português e inglês.
    ocr_text = pytesseract.image_to_string(
        image,
        lang="por+eng"
    ).strip()

    # Usa o OCR quando ele encontrar mais conteúdo.
    if len(ocr_text) > len(text):
        return ocr_text

    return text


def load_pdf(pdf_path: str) -> str:
    """
    Extrai o texto completo de um PDF digital ou digitalizado.
    Utiliza OCR automaticamente nas páginas sem texto suficiente.
    """

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {pdf_path}"
        )

    pages = []

    with fitz.open(str(path)) as document:
        for page_number, page in enumerate(document, start=1):
            text = extract_page_text(page)

            if text.strip():
                pages.append(
                    f"[Página {page_number}]\n{text}"
                )

    full_text = "\n\n".join(pages).strip()

    if not full_text:
        raise ValueError(
            "Não foi possível extrair texto do PDF, "
            "mesmo utilizando OCR. Verifique a qualidade "
            "das imagens e os idiomas instalados no Tesseract."
        )

    return full_text
