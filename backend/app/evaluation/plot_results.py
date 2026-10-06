import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CAMINHOS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]

SUMMARY_PATH = (
    BASE_DIR
    / "data"
    / "evaluation"
    / "benchmark_summary.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "evaluation"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "benchmark_metrics.png"
)


# ============================================================
# CARREGAR DADOS
# ============================================================

df = pd.read_csv(
    SUMMARY_PATH
)


# ============================================================
# GRÁFICO
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    df["metric"],
    df["mean"]
)

plt.ylim(
    0,
    1
)

plt.ylabel(
    "Valor médio"
)

plt.xlabel(
    "Métrica"
)

plt.title(
    "Avaliação do Retriever — Baseline"
)

plt.tight_layout()


# ============================================================
# SALVAR
# ============================================================

plt.savefig(
    OUTPUT_PATH,
    dpi=300
)

plt.show()

print(
    f"\nGráfico salvo em:\n{OUTPUT_PATH}"
)