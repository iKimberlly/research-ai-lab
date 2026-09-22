from backend.app.rag.loader import load_pdf


pdf_path = "data/documents/Reconhecimento+de+Emoções+como+ferramenta+de+apoio+às+terapias+personalizadas.pdf"

text = load_pdf(pdf_path)

print("\n===== TEXTO EXTRAÍDO =====\n")
print(text[:5000])