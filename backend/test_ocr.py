
from tkinter import Tk, filedialog

from app.rag.loader import load_pdf


root = Tk()
root.withdraw()

pdf_path = filedialog.askopenfilename(
    title="Selecione o artigo em PDF",
    filetypes=[("Arquivos PDF", "*.pdf")]
)

root.destroy()

if not pdf_path:
    print("Nenhum arquivo selecionado.")
    raise SystemExit(0)

try:
    text = load_pdf(pdf_path)

    print("\n===== RESULTADO DO OCR =====")
    print(f"Caracteres extraídos: {len(text)}")
    print("\n===== PRIMEIROS 2000 CARACTERES =====")
    print(text[:2000])

    output_path = "ocr_resultado.txt"

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(text)

    print(f"\nTexto salvo em: {output_path}")

except Exception as error:
    print(f"\nErro ao processar PDF: {error}")
