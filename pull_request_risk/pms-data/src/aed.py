"""
aed.py
Análise Exploratória de Dados (AED)
Predição de Mudança de Software (PMS)

- Unidade de análise: ARQUIVO
- Label: will_change (0 = não muda, 1 = muda)
- Dataset: pms_trf.csv (ou pms_transformed_complete.csv)

Executar:
  python3 aed.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# CONFIG
# =========================
DATA_PATH = "data/transformed/pms_transformed_complete.csv" # Qualquer coisa, altere aqui o caminho
OUTPUT_PREFIX = "aed_pms"

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 7)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)

print("=" * 80)
print("ANÁLISE EXPLORATÓRIA — PREDIÇÃO DE MUDANÇA DE SOFTWARE (PMS)")
print("=" * 80)

print("\nINFORMAÇÕES BÁSICAS DO DATASET:")
print(f"Formato: {df.shape[0]} linhas x {df.shape[1]} colunas")

print("\nCOLUNAS:")
print(", ".join(df.columns))

# =========================
# DADOS TEMPORAIS
# =========================
df["window_start"] = pd.to_datetime(df["window_start"], errors="coerce")
df["window_end"] = pd.to_datetime(df["window_end"], errors="coerce")

print("\nPERÍODO COBERTO:")
print(f"  Início: {df['window_start'].min().date()}")
print(f"  Fim:    {df['window_end'].max().date()}")

# =========================
# DISTRIBUIÇÃO DA LABEL
# =========================
print("\nDISTRIBUIÇÃO DA LABEL (will_change):")
label_dist = df["will_change"].value_counts(normalize=True) * 100
for k, v in label_dist.items():
    print(f"  will_change={k}: {v:.2f}%")

print(f"\nTaxa geral de arquivos que mudam: {df['will_change'].mean() * 100:.2f}%")

# =========================
# DISTRIBUIÇÃO POR REPOSITÓRIO
# =========================
print("\nDISTRIBUIÇÃO POR REPOSITÓRIO:")
repo_stats = df.groupby("repo").agg(
    total_files=("file", "count"),
    change_rate=("will_change", "mean")
).sort_values("total_files", ascending=False)

for repo, row in repo_stats.iterrows():
    print(f"{repo}: {row.total_files} arquivos | {row.change_rate*100:.1f}% mudam")

# =========================
# MÉTRICAS NUMÉRICAS
# =========================
numeric_cols = [
    "changes_count", "lines_added", "lines_removed", "churn",
    "complexity_sum", "n_authors"
]

print("\nESTATÍSTICAS DESCRITIVAS DAS MÉTRICAS:")
print(
    df[numeric_cols]
    .describe()
    .loc[["mean", "std", "min", "50%", "max"]]
    .round(2)
)

# =========================
# CORRELAÇÃO COM will_change
# =========================
print("\nCORRELAÇÃO DAS MÉTRICAS COM will_change:")
for col in numeric_cols:
    corr = df[[col, "will_change"]].corr().iloc[0, 1]
    print(f"  {col}: {corr:.3f}")

# =========================
# MATRIZ DE CORRELAÇÃO
# =========================
plt.figure(figsize=(10, 8))
corr_matrix = df[numeric_cols + ["will_change"]].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Matriz de Correlação — PMS")
plt.tight_layout()
plt.savefig(f"{OUTPUT_PREFIX}_correlation_matrix.png", dpi=300)
plt.show()

# =========================
# DISTRIBUIÇÃO DAS FEATURES
# =========================
for col in numeric_cols:
    plt.figure()
    sns.histplot(df[col], bins=50, kde=True)
    plt.title(f"Distribuição de {col}")
    plt.xlabel(col)
    plt.ylabel("Frequência")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_hist_{col}.png", dpi=300)
    plt.show()

# =========================
# COMPARAÇÃO: MUDA vs NÃO MUDA
# =========================
print("\nCOMPARAÇÃO ENTRE ARQUIVOS QUE MUDAM vs NÃO MUDAM (médias):")

comparison = []
for col in numeric_cols:
    mean_change = df[df["will_change"] == 1][col].mean()
    mean_no_change = df[df["will_change"] == 0][col].mean()
    diff_pct = ((mean_change - mean_no_change) / mean_no_change * 100) if mean_no_change > 0 else 0
    comparison.append([col, mean_change, mean_no_change, diff_pct])

comp_df = pd.DataFrame(
    comparison,
    columns=["Métrica", "Mudam", "Não mudam", "Diferença (%)"]
)

print(comp_df.round(2).to_string(index=False))

# =========================
# BOXPLOTS
# =========================
plt.figure(figsize=(14, 8))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x="will_change", y=col, data=df, showfliers=False)
    plt.title(col)
    plt.xlabel("will_change")
plt.tight_layout()
plt.savefig(f"{OUTPUT_PREFIX}_boxplots.png", dpi=300)
plt.show()

# =========================
# EVOLUÇÃO TEMPORAL
# =========================
df["year_month"] = df["window_start"].dt.to_period("M")

temporal = df.groupby("year_month").agg(
    n_files=("file", "count"),
    change_rate=("will_change", "mean")
).reset_index()

plt.figure(figsize=(14, 6))
ax1 = plt.gca()

ax1.plot(temporal["year_month"].astype(str), temporal["n_files"], "b-o", label="Arquivos")
ax1.set_ylabel("Número de arquivos", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")
ax1.tick_params(axis="x", rotation=45)

ax2 = ax1.twinx()
ax2.plot(
    temporal["year_month"].astype(str),
    temporal["change_rate"] * 100,
    "r-s",
    label="% Mudam"
)
ax2.set_ylabel("% de arquivos que mudam", color="red")
ax2.tick_params(axis="y", labelcolor="red")

plt.title("Evolução Temporal — PMS")
fig = plt.gcf()
fig.legend(loc="upper left")
plt.tight_layout()
plt.savefig(f"{OUTPUT_PREFIX}_temporal.png", dpi=300)
plt.show()

# =========================
# OUTLIERS (IQR)
# =========================
print("\nIDENTIFICAÇÃO DE OUTLIERS (IQR):")
for col in numeric_cols:
    data = df[col]
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    if IQR == 0:
        continue
    outliers = df[(data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)]
    if len(outliers) > 0:
        print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.2f}%)")

# =========================
# RESUMO FINAL
# =========================
print("\n" + "=" * 80)
print("RESUMO DA AED (PMS)")
print("=" * 80)
print(f"1. Total de registros: {len(df)}")
print(f"2. Repositórios analisados: {df['repo'].nunique()}")
print(f"3. Taxa média de mudança: {df['will_change'].mean()*100:.2f}%")
print(f"4. Feature mais correlacionada com mudança:")
best_corr = (
    df[numeric_cols + ["will_change"]]
    .corr()["will_change"]
    .drop("will_change")
    .abs()
    .sort_values(ascending=False)
)
print(f"   {best_corr.index[0]} (|corr|={best_corr.iloc[0]:.3f})")
print("=" * 80)

print("\nANÁLISE EXPLORATÓRIA CONCLUÍDA COM SUCESSO.")
