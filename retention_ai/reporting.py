from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")


def _finalize_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_class_balance(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=df, x="churn", hue="churn", palette=["#d5e7df", "#d8654f"], legend=False)
    ax.set_title("Distribution de la cible churn")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Nombre de clients")
    _finalize_plot(path)


def plot_contract_churn(df: pd.DataFrame, path: Path) -> None:
    summary = (
        df.groupby("contract_type", observed=False)["churn"]
        .mean()
        .sort_values(ascending=False)
        .reset_index(name="churn_rate")
    )
    plt.figure(figsize=(7, 4))
    ax = sns.barplot(
        data=summary,
        x="contract_type",
        y="churn_rate",
        hue="contract_type",
        palette="crest",
        legend=False,
    )
    ax.set_title("Taux de churn par type de contrat")
    ax.set_xlabel("Type de contrat")
    ax.set_ylabel("Taux de churn")
    _finalize_plot(path)


def plot_model_comparison(results_df: pd.DataFrame, path: Path) -> None:
    plot_df = results_df.melt(
        id_vars=["label"],
        value_vars=["test_pr_auc", "test_f1", "test_recall"],
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=plot_df, x="label", y="score", hue="metric", palette="mako")
    ax.set_title("Comparaison des modeles sur le jeu de test")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=20)
    _finalize_plot(path)


def plot_feature_importance(feature_df: pd.DataFrame, path: Path, top_n: int = 12) -> None:
    top_features = feature_df.head(top_n).sort_values("importance_mean")
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(
        data=top_features,
        x="importance_mean",
        y="feature",
        hue="feature",
        palette="viridis",
        legend=False,
    )
    ax.set_title("Variables les plus influentes (Permutation Importance)")
    ax.set_xlabel("Impact moyen sur la performance")
    ax.set_ylabel("")
    _finalize_plot(path)


def plot_confusion_matrix(metrics: dict[str, int], path: Path) -> None:
    matrix = pd.DataFrame(
        [
            [metrics["true_negative"], metrics["false_positive"]],
            [metrics["false_negative"], metrics["true_positive"]],
        ],
        index=["Reel 0", "Reel 1"],
        columns=["Pred 0", "Pred 1"],
    )
    plt.figure(figsize=(5, 4))
    ax = sns.heatmap(matrix, annot=True, fmt="d", cmap="YlGnBu")
    ax.set_title("Matrice de confusion du modele final")
    _finalize_plot(path)
