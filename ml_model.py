import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import BatchNormalization, Conv1D, Dense, Dropout, Flatten, MaxPool1D
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
from tensorflow.keras.optimizers import Adam


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def build_model(n_features: int) -> Sequential:
    model = Sequential()
    model.add(Conv1D(64, 2, activation="relu", input_shape=(n_features, 1)))
    model.add(BatchNormalization())
    model.add(MaxPool1D(2))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 2, activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPool1D(2))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def pick_best_threshold(y_true: np.ndarray, scores: np.ndarray, fallback: float = 0.5) -> tuple[float, float]:
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, scores)

    # precision_recall_curve returns one extra point compared to thresholds.
    if thresholds.size == 0:
        return fallback, 0.0

    f1_curve = (2 * precision_curve[:-1] * recall_curve[:-1]) / (
        precision_curve[:-1] + recall_curve[:-1] + 1e-12
    )
    best_idx = int(np.nanargmax(f1_curve))
    return float(thresholds[best_idx]), float(f1_curve[best_idx])


def evaluate_predictions(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_pred = (scores >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, digits=4),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parent
    dataset_path = project_root / "creditcard.csv"
    model_dir = project_root / "Model"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Please place creditcard.csv in the project root."
        )

    logger.info("Loading dataset from %s", dataset_path)
    data = pd.read_csv(dataset_path)

    class_distribution = data["Class"].value_counts().to_dict()
    logger.info("Class distribution: %s", class_distribution)

    X = data.drop(columns=["Class"])
    y = data["Class"].astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_all_scaled = scaler.transform(X)

    X_train_cnn = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
    X_test_cnn = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
    X_all_cnn = X_all_scaled.reshape((X_all_scaled.shape[0], X_all_scaled.shape[1], 1))

    model = build_model(X_train_cnn.shape[1])
    model.summary()

    logger.info("Training CNN model")
    model.fit(
        X_train_cnn,
        y_train,
        epochs=5,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
    )

    logger.info("Evaluating on holdout test split")
    test_scores = model.predict(X_test_cnn, verbose=0).ravel()
    test_pr_auc = float(average_precision_score(y_test, test_scores))
    test_roc_auc = float(roc_auc_score(y_test, test_scores))
    test_threshold, test_best_f1 = pick_best_threshold(y_test, test_scores, fallback=0.5)
    test_tuned = evaluate_predictions(y_test, test_scores, test_threshold)
    test_default = evaluate_predictions(y_test, test_scores, 0.5)

    logger.info(
        "Test Metrics | PR-AUC: %.4f | ROC-AUC: %.4f | Best threshold: %.6f | Best F1(PR curve): %.4f",
        test_pr_auc,
        test_roc_auc,
        test_threshold,
        test_best_f1,
    )
    logger.info(
        "Test (threshold-tuned) | Precision(1): %.4f | Recall(1): %.4f | F1(1): %.4f",
        test_tuned["precision"],
        test_tuned["recall"],
        test_tuned["f1"],
    )
    logger.info("Test Confusion Matrix (threshold-tuned):\n%s", np.array(test_tuned["confusion_matrix"]))
    logger.info("Test Classification Report (threshold-tuned):\n%s", test_tuned["classification_report"])
    logger.info(
        "Test (default threshold=0.5) | Precision(1): %.4f | Recall(1): %.4f | F1(1): %.4f",
        test_default["precision"],
        test_default["recall"],
        test_default["f1"],
    )
    logger.info("Test Confusion Matrix (default threshold=0.5):\n%s", np.array(test_default["confusion_matrix"]))
    logger.info("Test Classification Report (default threshold=0.5):\n%s", test_default["classification_report"])

    logger.info("Evaluating on full dataset for comparison against Isolation Forest output")
    all_scores = model.predict(X_all_cnn, verbose=0).ravel()
    all_pr_auc = float(average_precision_score(y, all_scores))
    all_roc_auc = float(roc_auc_score(y, all_scores))
    all_threshold, all_best_f1 = pick_best_threshold(y, all_scores, fallback=0.5)
    all_tuned = evaluate_predictions(y, all_scores, all_threshold)
    all_default = evaluate_predictions(y, all_scores, 0.5)

    logger.info(
        "Selected threshold from scores: %.6f (best F1 from PR curve: %.4f)",
        all_threshold,
        all_best_f1,
    )
    logger.info(
        "Metrics | PR-AUC: %.4f | ROC-AUC: %.4f | Precision(1): %.4f | Recall(1): %.4f | F1(1): %.4f",
        all_pr_auc,
        all_roc_auc,
        all_tuned["precision"],
        all_tuned["recall"],
        all_tuned["f1"],
    )
    logger.info("Confusion Matrix (threshold-tuned):\n%s", np.array(all_tuned["confusion_matrix"]))
    logger.info("Classification Report (threshold-tuned):\n%s", all_tuned["classification_report"])
    logger.info("Confusion Matrix (default threshold=0.5):\n%s", np.array(all_default["confusion_matrix"]))
    logger.info("Classification Report (default threshold=0.5):\n%s", all_default["classification_report"])

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "fraud_detection_model.keras"
    scaler_path = model_dir / "scaler.pkl"
    metadata_path = model_dir / "model_metadata_cnn.json"

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    metadata = {
        "model_type": "cnn_binary_classifier",
        "dataset_path": str(dataset_path),
        "test_metrics": {
            "pr_auc": test_pr_auc,
            "roc_auc": test_roc_auc,
            "best_threshold": test_threshold,
            "best_f1_from_pr_curve": test_best_f1,
            "threshold_tuned": {
                "precision": test_tuned["precision"],
                "recall": test_tuned["recall"],
                "f1": test_tuned["f1"],
            },
            "default_threshold_0_5": {
                "precision": test_default["precision"],
                "recall": test_default["recall"],
                "f1": test_default["f1"],
            },
        },
        "full_dataset_metrics": {
            "pr_auc": all_pr_auc,
            "roc_auc": all_roc_auc,
            "best_threshold": all_threshold,
            "best_f1_from_pr_curve": all_best_f1,
            "threshold_tuned": {
                "precision": all_tuned["precision"],
                "recall": all_tuned["recall"],
                "f1": all_tuned["f1"],
            },
            "default_threshold_0_5": {
                "precision": all_default["precision"],
                "recall": all_default["recall"],
                "f1": all_default["f1"],
            },
        },
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    logger.info("Saved model to %s", model_path)
    logger.info("Saved scaler to %s", scaler_path)
    logger.info("Saved metadata to %s", metadata_path)
    logger.info("Training and evaluation pipeline completed successfully")


if __name__ == "__main__":
    main()