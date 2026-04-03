from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def compute_classification_metrics(y_true, y_pred) -> Dict:
    """
    Compute core classification metrics.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }
    return metrics


def plot_confusion_matrix(y_true, y_pred, labels: List[str], save_path):
    """
    Save a confusion matrix figure.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(cm, aspect="auto")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(model, feature_names: List[str], save_path, top_n: int = 20):
    """
    Plot feature importances for tree-based models inside a pipeline.
    """
    estimator = None

    # Support sklearn pipeline
    if hasattr(model, "named_steps") and "classifier" in model.named_steps:
        estimator = model.named_steps["classifier"]
    else:
        estimator = model

    if not hasattr(estimator, "feature_importances_"):
        print("Model does not expose feature_importances_. Skipping importance plot.")
        return

    importances = estimator.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(importance_df["feature"][::-1], importance_df["importance"][::-1])
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
