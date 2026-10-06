import pandas as pd
from pathlib import Path


# ============================================================
# CAMINHOS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]

RESULTS_PATH = (
    BASE_DIR
    / "data"
    / "evaluation"
    / "benchmark_results.csv"
)

SUMMARY_PATH = (
    BASE_DIR
    / "data"
    / "evaluation"
    / "benchmark_summary.csv"
)


# ============================================================
# CARREGAR RESULTADOS
# ============================================================

df = pd.read_csv(
    RESULTS_PATH
)


print("\n" + "=" * 70)
print("ANÁLISE DO BENCHMARK")
print("=" * 70)


print(
    f"\nTotal de perguntas: {len(df)}"
)


# ============================================================
# MÉDIAS
# ============================================================

mean_precision = (
    df["precision_at_k"].mean()
)

mean_recall = (
    df["recall_at_k"].mean()
)

mean_mrr = (
    df["mrr"].mean()
)


print("\nMÉTRICAS MÉDIAS")
print("-" * 70)

print(
    f"Precision@K médio: {mean_precision:.3f}"
)

print(
    f"Recall@K médio:    {mean_recall:.3f}"
)

print(
    f"MRR médio:         {mean_mrr:.3f}"
)


# ============================================================
# RESUMO
# ============================================================

summary = pd.DataFrame([
    {
        "metric": "Precision@K",
        "mean": mean_precision
    },
    {
        "metric": "Recall@K",
        "mean": mean_recall
    },
    {
        "metric": "MRR",
        "mean": mean_mrr
    }
])


summary.to_csv(
    SUMMARY_PATH,
    index=False,
    encoding="utf-8"
)


print("\n" + "=" * 70)

print(
    f"Resumo salvo em:\n{SUMMARY_PATH}"
)

print("=" * 70)