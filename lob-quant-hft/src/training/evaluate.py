"""
Model Evaluation
================
Computes:
  - Accuracy, Precision, Recall, F1 (macro / weighted / per-class)
  - Matthews Correlation Coefficient
  - Cohen's Kappa
  - Confusion Matrix
  - ROC-AUC (OvR)
  - Calibration plots
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

CLASS_NAMES = ["Down", "Stationary", "Up"]


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
    return_probs: bool = False,
) -> dict:
    """
    Evaluate a model on a DataLoader.

    Returns
    -------
    dict with keys: loss, accuracy, precision, recall, f1, mcc, kappa,
                    confusion_matrix, report, roc_auc
                    [optionally: probs, preds, labels]
    """
    model.eval()
    all_logits, all_preds, all_labels = [], [], []
    total_loss = 0.0
    n = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)

        if criterion is not None:
            total_loss += criterion(logits, y).item() * y.size(0)
            n += y.size(0)

        all_logits.append(logits.cpu())
        all_preds.append(logits.argmax(-1).cpu())
        all_labels.append(y.cpu())

    logits_all = torch.cat(all_logits).numpy()
    preds_all  = torch.cat(all_preds).numpy()
    labels_all = torch.cat(all_labels).numpy()

    probs_all = _softmax(logits_all)

    # ── Metrics ─────────────────────────────────────────────────────────────
    acc    = accuracy_score(labels_all, preds_all)
    mcc    = matthews_corrcoef(labels_all, preds_all)
    kappa  = cohen_kappa_score(labels_all, preds_all)
    cm     = confusion_matrix(labels_all, preds_all)
    report = classification_report(
        labels_all, preds_all,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    try:
        roc_auc = roc_auc_score(
            labels_all, probs_all,
            multi_class="ovr", average="macro"
        )
    except ValueError:
        roc_auc = float("nan")

    result = {
        "loss":             total_loss / max(n, 1) if criterion else None,
        "accuracy":         float(acc),
        "precision_macro":  float(report["macro avg"]["precision"]),
        "recall_macro":     float(report["macro avg"]["recall"]),
        "f1_macro":         float(report["macro avg"]["f1-score"]),
        "f1_weighted":      float(report["weighted avg"]["f1-score"]),
        "mcc":              float(mcc),
        "kappa":            float(kappa),
        "roc_auc":          float(roc_auc),
        "confusion_matrix": cm,
        "report":           report,
        "per_class_f1": {
            CLASS_NAMES[i]: float(report[CLASS_NAMES[i]]["f1-score"])
            for i in range(len(CLASS_NAMES))
        },
    }

    if return_probs:
        result["probs"]  = probs_all
        result["preds"]  = preds_all
        result["labels"] = labels_all

    _log_results(result)
    return result


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _log_results(r: dict) -> None:
    logger.info(
        "Accuracy=%.4f  F1(macro)=%.4f  MCC=%.4f  Kappa=%.4f  ROC-AUC=%.4f",
        r["accuracy"], r["f1_macro"], r["mcc"], r["kappa"], r["roc_auc"],
    )
    logger.info("Per-class F1: %s", r["per_class_f1"])


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def print_confusion_matrix(cm: np.ndarray, class_names: list[str] | None = None) -> None:
    names = class_names or CLASS_NAMES
    header = f"{'':>12}" + "".join(f"{n:>12}" for n in names)
    print(header)
    for i, row in enumerate(cm):
        print(f"{names[i]:>12}" + "".join(f"{v:>12}" for v in row))


def print_report(report: dict) -> None:
    fmt = "{:>14}" + "{:>10}" * 4
    print(fmt.format("", "precision", "recall", "f1-score", "support"))
    for cls in CLASS_NAMES:
        r = report[cls]
        print(fmt.format(
            cls, f"{r['precision']:.4f}", f"{r['recall']:.4f}",
            f"{r['f1-score']:.4f}", int(r["support"]),
        ))
    for avg in ("macro avg", "weighted avg"):
        r = report[avg]
        print(fmt.format(
            avg, f"{r['precision']:.4f}", f"{r['recall']:.4f}",
            f"{r['f1-score']:.4f}", int(r["support"]),
        ))
