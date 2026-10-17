
import pymupdf
from tkinter import Tk, filedialog


root = Tk()
root.withdraw()

pdf_path = filedialog.askopenfilename(
    title="Selecione o artigo que falhou no upload",
    filetypes=[("Arquivos PDF", "*.pdf")]
)

root.destroy()

if not pdf_path:
    print("Nenhum arquivo selecionado.")
    raise SystemExit(0)

document = pymupdf.open(pdf_path)

total_chars = 0
total_images = 0

print(f"\nArquivo: {pdf_path}")
print(f"Total de páginas: {len(document)}")

for page_number, page in enumerate(document, start=1):
    text = page.get_text("text")
    images = page.get_images(full=True)

    total_chars += len(text.strip())
    total_images += len(images)

    print(
        f"Página {page_number}: "
        f"{len(text.strip())} caracteres | "
        f"{len(images)} imagens"
    )

print("\n===== RESUMO =====")
print(f"Caracteres extraídos: {total_chars}")
print(f"Imagens encontradas: {total_images}")

document.close()
