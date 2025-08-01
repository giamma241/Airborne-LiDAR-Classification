import logging

import numpy as np
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s"
)


def evaluate_classification(pred_labels, true_labels, label_names=None):
    """
    Evaluates classification results against ground truth.

    Params:
    - pred_labels: 1D array-like of predicted class labels
    - true_labels: 1D array-like of ground truth labels
    - label_names: dict mapping label int → str (optional)

    Returns:
    - dict with accuracy, precision, recall, f1-score, IoU
    """
    logging.info("Evaluating classification performance.")

    metrics = {
        "accuracy": np.mean(pred_labels == true_labels),
        "precision": precision_score(
            true_labels, pred_labels, average="weighted", zero_division=0
        ),
        "recall": recall_score(
            true_labels, pred_labels, average="weighted", zero_division=0
        ),
        "f1_score": f1_score(
            true_labels, pred_labels, average="weighted", zero_division=0
        ),
        "iou": jaccard_score(
            true_labels, pred_labels, average="weighted", zero_division=0
        ),
    }

    logging.info("Accuracy: %.3f", metrics["accuracy"])
    logging.info("Precision: %.3f", metrics["precision"])
    logging.info("Recall: %.3f", metrics["recall"])
    logging.info("F1 Score: %.3f", metrics["f1_score"])
    logging.info("IoU: %.3f", metrics["iou"])

    return metrics
