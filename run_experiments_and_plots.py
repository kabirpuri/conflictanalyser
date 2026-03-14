#!/usr/bin/env python3
"""
Run multiple model tests and generate report-ready plots.

This script builds on `conflict_model_pipeline.py` and produces:
- Cross-validation metrics for multiple models
- Ablation tests (categorical-only, text-only, combined)
- Confusion matrix plot (best model)
- Class distribution plot
- Top feature importance plot (Random Forest)
- Model comparison plot
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Allow local vendored dependencies in constrained environments.
VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

import matplotlib.pyplot as plt
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OneHotEncoder

from conflict_model_pipeline import load_and_clean_survey


SURVEY_FILE = Path("Cleaned_Construction_Conflict_Survey_Data.csv")
STATS_FILE = Path("Conflict_Factor_Statistics_Final.csv")
OUTPUT_DIR = Path("outputs/plots")
TABLE_DIR = Path("outputs")


def build_feature_matrices(df: pd.DataFrame):
    model_df = df[df["Conflict_Severity_Num"].notna()].copy()
    model_df["Conflict_Severity_Num"] = model_df["Conflict_Severity_Num"].astype(int)

    categorical_cols = ["Role", "Experience", "Project_Sector"]
    onehot = OneHotEncoder(handle_unknown="ignore")
    X_cat = onehot.fit_transform(model_df[categorical_cols].fillna("Unknown"))

    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=250,
    )
    X_text = tfidf.fit_transform(model_df["text_blob"])
    X_combined = hstack([X_cat, X_text])
    y = model_df["Conflict_Severity_Num"].values

    return model_df, X_cat, X_text, X_combined, y, onehot, tfidf


def cv_scores(model, X, y, cv):
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={"accuracy": "accuracy", "f1_weighted": "f1_weighted"},
        n_jobs=1,
        return_train_score=False,
    )
    return {
        "accuracy_mean": float(np.mean(scores["test_accuracy"])),
        "accuracy_std": float(np.std(scores["test_accuracy"])),
        "f1_weighted_mean": float(np.mean(scores["test_f1_weighted"])),
        "f1_weighted_std": float(np.std(scores["test_f1_weighted"])),
    }


def plot_class_distribution(model_df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    counts = model_df["Conflict_Severity_Num"].value_counts().sort_index()
    label_map = {1: "Low", 2: "Medium", 3: "High"}
    labels = [label_map.get(i, str(i)) for i in counts.index]

    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(labels, counts.values)
    plt.title("Conflict Severity Class Distribution")
    plt.xlabel("Severity Class")
    plt.ylabel("Count")
    for b in bars:
        plt.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.2,
            f"{int(b.get_height())}",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=300)
    plt.close()


def plot_model_comparison(results_df: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 5))
    x = np.arange(len(results_df))
    width = 0.38
    plt.bar(x - width / 2, results_df["accuracy_mean"], width=width, label="Accuracy")
    plt.bar(x + width / 2, results_df["f1_weighted_mean"], width=width, label="Weighted F1")
    plt.xticks(x, results_df["test_name"], rotation=35, ha="right")
    plt.ylim(0, 1)
    plt.title("Model/Test Comparison (Repeated Stratified CV)")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=300)
    plt.close()


def plot_confusion_matrix_best_rf(X_combined, y) -> Dict[str, float]:
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.25, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    cm = confusion_matrix(y_test, pred, labels=[1, 2, 3])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low", "Medium", "High"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix (Random Forest)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix_rf.png", dpi=300)
    plt.close(fig)

    return {
        "holdout_accuracy": float(accuracy_score(y_test, pred)),
        "holdout_weighted_f1": float(f1_score(y_test, pred, average="weighted")),
    }


def plot_top_feature_importance_rf(X_combined, y, onehot, tfidf) -> None:
    model = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2,
    )
    model.fit(X_combined, y)
    feature_names = list(onehot.get_feature_names_out(["Role", "Experience", "Project_Sector"])) + list(
        tfidf.get_feature_names_out()
    )
    imp = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_}).sort_values(
        "importance", ascending=False
    )
    top_n = imp.head(15).iloc[::-1]

    plt.figure(figsize=(9, 6))
    plt.barh(top_n["feature"], top_n["importance"])
    plt.title("Top 15 Feature Importances (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top15_feature_importance_rf.png", dpi=300)
    plt.close()

    imp.head(30).to_csv(TABLE_DIR / "rf_feature_importance_top30_experiments.csv", index=False)


def plot_conflict_factor_stats_if_available() -> List[str]:
    if not STATS_FILE.exists():
        return []

    stats = pd.read_csv(STATS_FILE).sort_values("Rank", ascending=True)

    plt.figure(figsize=(9, 6))
    plt.barh(stats["Conflict_Factor"][::-1], stats["Mean"][::-1])
    plt.xlabel("Mean Score")
    plt.title("Conflict Factor Ranking (Descriptive Statistics)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "conflict_factor_ranking.png", dpi=300)
    plt.close()

    stats["Normalized_Weight"] = stats["Mean"] / stats["Mean"].sum()
    weights = stats.sort_values("Normalized_Weight", ascending=False)
    plt.figure(figsize=(9, 6))
    plt.bar(weights["Conflict_Factor"], weights["Normalized_Weight"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Normalized Weight")
    plt.title("Conflict Factor Normalized Weights")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "conflict_factor_weights.png", dpi=300)
    plt.close()

    return [
        "outputs/plots/conflict_factor_ranking.png",
        "outputs/plots/conflict_factor_weights.png",
    ]


def main() -> None:
    if not SURVEY_FILE.exists():
        raise FileNotFoundError(f"Missing file: {SURVEY_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_clean_survey(SURVEY_FILE)
    model_df, X_cat, X_text, X_combined, y, onehot, tfidf = build_feature_matrices(df)
    plot_class_distribution(model_df)

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

    tests: List[Tuple[str, object, object]] = [
        (
            "RF_combined",
            RandomForestClassifier(
                n_estimators=500, random_state=42, class_weight="balanced", min_samples_leaf=2
            ),
            X_combined,
        ),
        (
            "RF_text_only",
            RandomForestClassifier(
                n_estimators=500, random_state=42, class_weight="balanced", min_samples_leaf=2
            ),
            X_text,
        ),
        (
            "RF_categorical_only",
            RandomForestClassifier(
                n_estimators=500, random_state=42, class_weight="balanced", min_samples_leaf=2
            ),
            X_cat,
        ),
        ("LogReg_combined", LogisticRegression(max_iter=3000, class_weight="balanced"), X_combined),
        ("MultinomialNB_text_only", MultinomialNB(alpha=0.5), X_text),
    ]

    rows: List[Dict[str, float]] = []
    for name, model, X in tests:
        out = cv_scores(model, X, y, cv)
        out["test_name"] = name
        rows.append(out)

    results_df = pd.DataFrame(rows)[
        ["test_name", "accuracy_mean", "accuracy_std", "f1_weighted_mean", "f1_weighted_std"]
    ].sort_values("f1_weighted_mean", ascending=False)
    results_df.to_csv(TABLE_DIR / "model_test_results_cv.csv", index=False)
    plot_model_comparison(results_df)

    holdout_metrics = plot_confusion_matrix_best_rf(X_combined, y)
    plot_top_feature_importance_rf(X_combined, y, onehot, tfidf)
    stats_plots = plot_conflict_factor_stats_if_available()

    summary = {
        "n_rows_used": int(len(model_df)),
        "n_classes": int(len(np.unique(y))),
        "best_by_f1_weighted_cv": results_df.iloc[0].to_dict(),
        "rf_holdout_metrics": holdout_metrics,
        "plots_generated": [
            "outputs/plots/class_distribution.png",
            "outputs/plots/model_comparison.png",
            "outputs/plots/confusion_matrix_rf.png",
            "outputs/plots/top15_feature_importance_rf.png",
        ]
        + stats_plots,
    }

    (TABLE_DIR / "experiments_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Experiment suite complete.")
    print(f"Best CV test: {summary['best_by_f1_weighted_cv']['test_name']}")
    print(f"Plots saved under: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
