import logging
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CONTAMINATION = 0.002
CONTAMINATION_CANDIDATES = [0.001, 0.0015, 0.002, 0.003, 0.005]
VALIDATION_SIZE = 0.2
RANDOM_STATE = 42
DATA_PATH = Path("creditcard.csv")
MODEL_DIR = Path("./Model")
MODEL_PATH = MODEL_DIR / "anomaly_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _safe_roc_auc(y_true: pd.Series, fraud_scores: np.ndarray) -> float:
    if y_true.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, fraud_scores))


def _best_threshold_from_scores(y_true: pd.Series, fraud_scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, fraud_scores)
    if len(thresholds) == 0:
        return 0.0, 0.0

    # precision/recall arrays have one extra point vs thresholds.
    f1_values = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = int(np.nanargmax(f1_values))
    return float(thresholds[best_idx]), float(f1_values[best_idx])


def _evaluate_with_threshold(
    y_true: pd.Series,
    fraud_scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_pred = (fraud_scores >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision_fraud": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_fraud": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_fraud": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, fraud_scores)),
        "roc_auc": _safe_roc_auc(y_true, fraud_scores),
    }


def _confusion_as_dict(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path.resolve()}")

    data = pd.read_csv(csv_path)
    data.columns = data.columns.str.strip().str.strip('"')

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    # Keep only model-relevant columns in a deterministic order.
    data = data[required_columns].copy()

    for column in required_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=required_columns)
    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(int)

    unexpected_labels = set(data[TARGET_COLUMN].unique()) - {0, 1}
    if unexpected_labels:
        raise ValueError(
            f"{TARGET_COLUMN} must contain only 0 (normal) and 1 (fraud). Found: {sorted(unexpected_labels)}"
        )

    return data


def tune_contamination(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[float, list[dict[str, float]]]:
    X_train_normal = X_train[y_train == 0]
    if X_train_normal.empty:
        raise ValueError("No normal transactions found in training split.")

    scaler = StandardScaler()
    X_train_normal_scaled = scaler.fit_transform(X_train_normal)
    X_val_scaled = scaler.transform(X_val)

    candidate_results: list[dict[str, float]] = []
    for contamination in CONTAMINATION_CANDIDATES:
        model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train_normal_scaled)

        fraud_scores = -model.decision_function(X_val_scaled)
        threshold, _ = _best_threshold_from_scores(y_val, fraud_scores)
        metrics = _evaluate_with_threshold(y_val, fraud_scores, threshold)
        metrics["contamination"] = float(contamination)
        candidate_results.append(metrics)

    best_result = max(candidate_results, key=lambda item: (item["pr_auc"], item["f1_fraud"]))
    return float(best_result["contamination"]), candidate_results


def train_final_model(
    data: pd.DataFrame,
    contamination: float,
) -> tuple[IsolationForest, StandardScaler, np.ndarray, pd.Series]:
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    X_train_normal = X[y == 0]
    if X_train_normal.empty:
        raise ValueError("No normal transactions found. Expected rows where Class == 0.")

    scaler = StandardScaler()
    X_train_normal_scaled = scaler.fit_transform(X_train_normal)
    X_scaled = scaler.transform(X)

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_normal_scaled)

    return model, scaler, X_scaled, y


def evaluate_model(model: IsolationForest, X_scaled: np.ndarray, y_true: pd.Series) -> dict[str, object]:
    fraud_scores = -model.decision_function(X_scaled)
    threshold, best_f1 = _best_threshold_from_scores(y_true, fraud_scores)
    threshold_metrics = _evaluate_with_threshold(y_true, fraud_scores, threshold)

    y_pred_threshold = (fraud_scores >= threshold).astype(int)
    y_pred_default = pd.Series(model.predict(X_scaled)).map({-1: 1, 1: 0}).to_numpy()

    logging.info("Selected threshold from scores: %.6f (best F1 from PR curve: %.4f)", threshold, best_f1)
    logging.info(
        "Metrics | PR-AUC: %.4f | ROC-AUC: %.4f | Precision(1): %.4f | Recall(1): %.4f | F1(1): %.4f",
        threshold_metrics["pr_auc"],
        threshold_metrics["roc_auc"],
        threshold_metrics["precision_fraud"],
        threshold_metrics["recall_fraud"],
        threshold_metrics["f1_fraud"],
    )

    logging.info("Confusion Matrix (threshold-tuned):\n%s", confusion_matrix(y_true, y_pred_threshold))
    logging.info(
        "Classification Report (threshold-tuned):\n%s",
        classification_report(y_true, y_pred_threshold, digits=4),
    )

    logging.info("Confusion Matrix (model default labels):\n%s", confusion_matrix(y_true, y_pred_default))
    logging.info(
        "Classification Report (model default labels):\n%s",
        classification_report(y_true, y_pred_default, digits=4),
    )

    return {
        "selected_threshold": float(threshold),
        "threshold_metrics": threshold_metrics,
        "threshold_confusion": _confusion_as_dict(y_true, y_pred_threshold),
        "default_confusion": _confusion_as_dict(y_true, y_pred_default),
    }


def save_artifacts(
    model: IsolationForest,
    scaler: StandardScaler,
    metadata: dict[str, object],
) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logging.info("Saved model to %s", MODEL_PATH.resolve())
    logging.info("Saved scaler to %s", SCALER_PATH.resolve())
    logging.info("Saved metadata to %s", METADATA_PATH.resolve())


def main() -> None:
    configure_logging()
    try:
        logging.info("Loading dataset from %s", DATA_PATH.resolve())
        data = load_dataset(DATA_PATH)

        X = data[FEATURE_COLUMNS]
        y = data[TARGET_COLUMN]
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=VALIDATION_SIZE,
            stratify=y,
            random_state=RANDOM_STATE,
        )

        logging.info("Tuning contamination on validation split")
        best_contamination, candidate_results = tune_contamination(X_train, y_train, X_val, y_val)
        logging.info("Best contamination selected: %.4f", best_contamination)
        for result in candidate_results:
            logging.info(
                "Candidate contamination %.4f | PR-AUC: %.4f | F1(1): %.4f | Recall(1): %.4f | Precision(1): %.4f",
                result["contamination"],
                result["pr_auc"],
                result["f1_fraud"],
                result["recall_fraud"],
                result["precision_fraud"],
            )

        logging.info("Training final Isolation Forest using only normal transactions (Class == 0)")
        model, scaler, X_scaled, y_true = train_final_model(data, contamination=best_contamination)

        logging.info("Evaluating on full dataset")
        eval_summary = evaluate_model(model, X_scaled, y_true)

        metadata = {
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "random_state": RANDOM_STATE,
            "contamination": best_contamination,
            "validation_size": VALIDATION_SIZE,
            "contamination_candidates": CONTAMINATION_CANDIDATES,
            "selected_threshold": eval_summary["selected_threshold"],
            "threshold_metrics": eval_summary["threshold_metrics"],
            "threshold_confusion": eval_summary["threshold_confusion"],
            "default_confusion": eval_summary["default_confusion"],
        }

        save_artifacts(model, scaler, metadata)
        logging.info("Training pipeline completed successfully")
    except Exception as exc:
        logging.exception("Training failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
