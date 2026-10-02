def precision_at_k(
    retrieved_chunks: list[int],
    relevant_chunks: set[int],
    k: int
) -> float:

    retrieved = retrieved_chunks[:k]

    if not retrieved:
        return 0.0

    relevant = sum(
        1 for chunk in retrieved
        if chunk in relevant_chunks
    )

    return relevant / len(retrieved)


def recall_at_k(
    retrieved_chunks: list[int],
    relevant_chunks: set[int],
    k: int
) -> float:

    retrieved = retrieved_chunks[:k]

    if not relevant_chunks:
        return 0.0

    relevant = sum(
        1 for chunk in retrieved
        if chunk in relevant_chunks
    )

    return relevant / len(relevant_chunks)


def reciprocal_rank(
    retrieved_chunks: list[int],
    relevant_chunks: set[int]
) -> float:

    for rank, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        if chunk in relevant_chunks:
            return 1 / rank

    return 0.0


def mean_reciprocal_rank(
    results: list[float]
) -> float:

    if not results:
        return 0.0

    return sum(results) / len(results)