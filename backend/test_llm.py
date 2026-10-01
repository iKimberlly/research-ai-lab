from app.llm.client import LLMClient


llm = LLMClient()


prompt = """
Explique de forma acadêmica e objetiva
o que é reconhecimento de emoções
por meio de expressões faciais.
"""


response = llm.generate(prompt)


print("\n===== RESPOSTA DA LLM =====\n")
print(response)