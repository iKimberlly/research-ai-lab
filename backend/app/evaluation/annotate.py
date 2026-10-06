import json
from pathlib import Path

from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore


# ============================================================
# DIRETÓRIOS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]

QUESTIONS_PATH = (
    BASE_DIR
    / "data"
    / "evaluation"
    / "questions.json"
)


# ============================================================
# CARREGAR PERGUNTAS
# ============================================================

def load_questions():

    with open(
        QUESTIONS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# SALVAR PERGUNTAS
# ============================================================

def save_questions(questions):

    with open(
        QUESTIONS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            questions,
            file,
            ensure_ascii=False,
            indent=4
        )


# ============================================================
# ANOTAÇÃO
# ============================================================

def annotate():

    vector_store = VectorStore()

    retriever = Retriever(
        vector_store
    )

    questions = load_questions()

    print("\n")
    print("=" * 80)
    print("ANOTAÇÃO DO GROUND TRUTH")
    print("=" * 80)

    print(
        "\nPara cada chunk, responda:"
    )

    print(
        "S = relevante"
    )

    print(
        "N = não relevante"
    )

    print(
        "A = parar a anotação da pergunta"
    )

    for item in questions:

        question = item["question"]

        print("\n")
        print("=" * 80)
        print("PERGUNTA")
        print("=" * 80)

        print(question)

        # ----------------------------------------------------
        # Buscar chunks
        # ----------------------------------------------------

        results = retriever.search(
            query=question,
            top_k=3
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        relevant_chunks = []

        # ----------------------------------------------------
        # Mostrar chunks
        # ----------------------------------------------------

        for rank, (
            document,
            metadata,
            distance
        ) in enumerate(
            zip(
                documents,
                metadatas,
                distances
            ),
            start=1
        ):

            chunk_index = metadata["chunk_index"]

            print("\n")
            print("-" * 80)

            print(
                f"RANKING: #{rank}"
            )

            print(
                f"CHUNK: {chunk_index}"
            )

            print(
                f"DISTÂNCIA: {distance:.4f}"
            )

            print("\nTRECHO:")

            print(document)

            print("\n" + "-" * 80)

            while True:

                answer = input(
                    "\nEsse chunk é relevante? "
                    "[S/N/A]: "
                ).strip().upper()

                if answer in ["S", "N", "A"]:
                    break

                print(
                    "Digite S, N ou A."
                )

            if answer == "S":

                relevant_chunks.append(
                    chunk_index
                )

            elif answer == "A":

                break

        # ----------------------------------------------------
        # Atualizar ground truth
        # ----------------------------------------------------

        item["relevant_chunks"] = sorted(
            relevant_chunks
        )

        print("\n")
        print("=" * 80)

        print(
            "Chunks marcados como relevantes:"
        )

        print(
            item["relevant_chunks"]
        )

        print("=" * 80)

        save_questions(
            questions
        )

        print(
            "\nGround truth salvo."
        )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    annotate()