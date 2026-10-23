
from sentence_transformers import SentenceTransformer


MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    def encode(self, texts: list[str]):
        return self.model.encode(
            texts,
            normalize_embeddings=True
        )
