from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200
) -> List[str]:
    """
    Divide um texto em chunks utilizando tamanho
    máximo e sobreposição entre os segmentos.

    Args:
        text: Texto completo do documento.
        chunk_size: Número máximo aproximado de caracteres.
        overlap: Quantidade de caracteres compartilhados
                 entre chunks consecutivos.

    Returns:
        Lista de chunks.
    """

    if not text.strip():
        return []

    if overlap >= chunk_size:
        raise ValueError(
            "overlap deve ser menor que chunk_size"
        )

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks