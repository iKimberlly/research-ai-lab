from pathlib import Path

import fitz


def load_pdf(pdf_path: str) -> str:
    """
    Extrai o texto completo de um arquivo PDF.
    """

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {pdf_path}"
        )

    document = fitz.open(pdf_path)

    pages = []

    for page in document:
        text = page.get_text()

        if text.strip():
            pages.append(text)

    document.close()

    return "\n".join(pages)