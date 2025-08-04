"""
Model evaluation utilities
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def evaluate_regression_model(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate regression model performance

    Args:
        y_true: True target values
        y_pred: Predicted target values

    Returns:
        Dictionary with evaluation metrics
    """
    metrics = {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }

    # Calculate additional metrics
    metrics["mape"] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    metrics["max_error"] = np.max(np.abs(y_true - y_pred))

    logger.info(f"Evaluation metrics: {metrics}")
    return metrics


def calculate_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate prediction residuals

    Args:
        y_true: True target values
        y_pred: Predicted target values

    Returns:
        Array of residuals
    """
    return y_true - y_pred


def plot_predictions_vs_actual(
    y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None
) -> None:
    """
    Plot predictions vs actual values

    Args:
        y_true: True target values
        y_pred: Predicted target values
        save_path: Path to save plot (optional)
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot(
        [y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2
    )
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Predictions vs Actual Values")
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_residuals(
    y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None
) -> None:
    """
    Plot prediction residuals

    Args:
        y_true: True target values
        y_pred: Predicted target values
        save_path: Path to save plot (optional)
    """
    residuals = calculate_residuals(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Residuals vs predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(y=0, color="r", linestyle="--")
    axes[0].set_xlabel("Predicted Values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Predicted Values")
    axes[0].grid(True, alpha=0.3)

    # Residuals histogram
    axes[1].hist(residuals, bins=50, alpha=0.7, edgecolor="black")
    axes[1].set_xlabel("Residuals")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Residuals Distribution")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def model_performance_summary(
    y_true: np.ndarray, y_pred: np.ndarray
) -> pd.DataFrame:
    """
    Generate comprehensive model performance summary

    Args:
        y_true: True target values
        y_pred: Predicted target values

    Returns:
        DataFrame with performance summary
    """
    metrics = evaluate_regression_model(y_true, y_pred)
    residuals = calculate_residuals(y_true, y_pred)

    summary_data = {
        "Metric": list(metrics.keys()) + ["std_residuals", "skew_residuals"],
        "Value": list(metrics.values())
        + [np.std(residuals), pd.Series(residuals).skew()],
    }

    summary_df = pd.DataFrame(summary_data)
    return summary_df


def cross_validation_evaluation(
    model, X: np.ndarray, y: np.ndarray, cv: int = 5
) -> Dict[str, Any]:
    """
    Perform cross-validation evaluation

    Args:
        model: Trained model
        X: Feature matrix
        y: Target vector
        cv: Number of cross-validation folds

    Returns:
        Dictionary with cross-validation results
    """
    from sklearn.model_selection import cross_val_score

    # Perform cross-validation for different metrics
    scoring = ["neg_mean_squared_error", "neg_mean_absolute_error", "r2"]
    cv_results = {}

    for score in scoring:
        scores = cross_val_score(model, X, y, cv=cv, scoring=score)
        metric_name = score.replace("neg_", "").replace("_", " ").title()
        cv_results[f"{metric_name} Mean"] = scores.mean()
        cv_results[f"{metric_name} Std"] = scores.std()

    logger.info(f"Cross-validation results: {cv_results}")
    return cv_results
