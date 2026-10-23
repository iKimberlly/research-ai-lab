
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

questions = [
    "O que é gamificação?",
    "What is gamification?",
    "Quais são os efeitos da gamificação nas emoções dos estudantes?",
    "What are the effects of gamification on students' emotions?",
]

texts = [
    "Gamification elements are used in educational activities.",
    "The study found no statistically significant differences "
    "in emotional expressions between gamified and non-gamified conditions.",
    "The study investigates the effects of gamification on "
    "students' basic emotions.",
]

question_embeddings = model.encode(
    questions,
    normalize_embeddings=True
)

text_embeddings = model.encode(
    texts,
    normalize_embeddings=True
)

for question, embedding in zip(questions, question_embeddings):
    similarities = text_embeddings @ embedding

    print(f"\nPergunta: {question}")

    for index in similarities.argsort()[::-1]:
        print(
            f"Similaridade: {similarities[index]:.4f} | "
            f"Texto: {texts[index]}"
        )
