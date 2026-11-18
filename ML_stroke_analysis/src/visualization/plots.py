"""Visualisasi EDA & evaluasi."""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

from src.config import FIGURES_DIR
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_target_distribution(df, target="stroke"):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=target, data=df, ax=ax)
    ax.set_title("Distribusi Target (Stroke)")
    fig.savefig(FIGURES_DIR / "target_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlation(df):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(),
                annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    fig.savefig(FIGURES_DIR / "correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr(y_true, y_prob, tag="model"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr, tpr, label="ROC")
    axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].set_title("ROC Curve"); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].legend()

    axes[1].plot(rec, prec, label="PR")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].legend()

    fig.savefig(FIGURES_DIR / f"{tag}_roc_pr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(y_true, y_pred, tag="model"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.savefig(FIGURES_DIR / f"{tag}_confusion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)