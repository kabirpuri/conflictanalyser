#!/usr/bin/env python3
"""
End-to-end modeling pipeline for construction conflict severity.

This script is designed for the currently available CSV files in this repo:
1) Cleaned_Construction_Conflict_Survey_Data.csv
2) Conflict_Factor_Statistics_Final.csv

It performs:
- Robust cleaning of survey records
- Conflict severity target normalization
- Text + demographic feature engineering
- Random Forest classification
- Export of metrics and feature importance

Usage:
    python3 conflict_model_pipeline.py
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Allow local vendored dependencies in constrained environments.
VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


SURVEY_FILE = Path("Cleaned_Construction_Conflict_Survey_Data.csv")
STATS_FILE = Path("Conflict_Factor_Statistics_Final.csv")
OUTPUT_DIR = Path("outputs")

LIKERT_COLUMNS = [
    "Payment_Delay",
    "Cost_Estimation_Error",
    "Material_Cost_Escalation",
    "Design_Ambiguity",
    "Poor_Workmanship",
    "Site_Conditions",
    "Leadership_Issues",
    "Trust_Deficit",
    "Unrealistic_Expectations",
    "Contract_Ambiguity",
    "Contract_Rigidity",
]


@dataclass
class DatasetQuality:
    total_rows: int
    usable_rows: int
    likert_non_null_ratio: float


def _normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def map_conflict_severity(value: object) -> float:
    """
    Map mixed severity values into ordered numeric classes:
    1 = Low, 2 = Medium, 3 = High

    Accepts either:
    - raw numeric strings (1..5) from later responses
    - descriptive severity labels from early responses
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan

    text = str(value).strip().lower()

    if text in {"1", "2", "3", "4", "5"}:
        n = int(text)
        # Collapsed bins to keep class balance workable in a small dataset.
        if n <= 2:
            return 1
        if n == 3:
            return 2
        return 3

    if "low / none" in text or "low" in text:
        return 1
    if "medium" in text:
        return 2
    if "high / critical" in text or "high" in text or "critical" in text:
        return 3

    return np.nan


def load_and_clean_survey(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # The first column in this dataset is a long consent statement.
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "Consent"})

    for col in [
        "Role",
        "Experience",
        "Project_Sector",
        "Trigger_Event",
        "Conflict_Progression",
        "Hidden_Factors",
        "Additional_Comments",
        "Conflict_Severity",
    ]:
        if col in df.columns:
            df[col] = df[col].map(_normalize_text)

    df["Conflict_Severity_Label"] = df["Conflict_Severity"]
    df["Conflict_Severity_Num"] = df["Conflict_Severity"].map(map_conflict_severity)

    # Merge all qualitative context into a single analyzable text field.
    text_cols = [
        "Trigger_Event",
        "Conflict_Progression",
        "Hidden_Factors",
        "Additional_Comments",
    ]
    df["text_blob"] = (
        df[text_cols]
        .fillna("")
        .agg(" ".join, axis=1)
        .map(_normalize_text)
        .replace("", "no_response")
    )

    return df


def dataset_quality_report(df: pd.DataFrame) -> DatasetQuality:
    non_null_likert = df[LIKERT_COLUMNS].notna().sum().sum()
    total_likert_cells = len(df) * len(LIKERT_COLUMNS)
    ratio = (non_null_likert / total_likert_cells) if total_likert_cells else 0.0

    usable_rows = df["Conflict_Severity_Num"].notna().sum()
    return DatasetQuality(
        total_rows=len(df),
        usable_rows=int(usable_rows),
        likert_non_null_ratio=float(ratio),
    )


def train_random_forest(df: pd.DataFrame) -> Dict[str, object]:
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

    X = hstack([X_cat, X_text])
    y = model_df["Conflict_Severity_Num"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    clf = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    cat_feature_names = list(onehot.get_feature_names_out(categorical_cols))
    text_feature_names = list(tfidf.get_feature_names_out())
    all_feature_names = cat_feature_names + text_feature_names

    importances = pd.DataFrame(
        {
            "feature": all_feature_names,
            "importance": clf.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    report = classification_report(y_test, y_pred, digits=4)

    return {
        "accuracy": float(accuracy),
        "weighted_f1": float(weighted_f1),
        "classification_report": report,
        "importances": importances,
    }


def export_outputs(df: pd.DataFrame, quality: DatasetQuality, model_out: Dict[str, object]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_export_cols = [
        "Consent",
        "Role",
        "Experience",
        "Project_Sector",
        "Trigger_Event",
        "Conflict_Progression",
        "Hidden_Factors",
        "Additional_Comments",
        "Conflict_Severity_Label",
        "Conflict_Severity_Num",
        "text_blob",
    ]
    df[cleaned_export_cols].to_csv(OUTPUT_DIR / "survey_cleaned_for_model.csv", index=False)

    metrics = {
        "total_rows": quality.total_rows,
        "usable_rows": quality.usable_rows,
        "likert_non_null_ratio": round(quality.likert_non_null_ratio, 4),
        "model_accuracy": round(model_out["accuracy"], 4),
        "model_weighted_f1": round(model_out["weighted_f1"], 4),
    }
    (OUTPUT_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "classification_report.txt").write_text(
        model_out["classification_report"], encoding="utf-8"
    )

    model_out["importances"].head(30).to_csv(
        OUTPUT_DIR / "top_feature_importance.csv",
        index=False,
    )

    # Keep the factor-level stats table as a reference (already computed in your previous workflow).
    if STATS_FILE.exists():
        stats = pd.read_csv(STATS_FILE)
        stats["Normalized_Weight"] = stats["Mean"] / stats["Mean"].sum()
        stats.to_csv(OUTPUT_DIR / "conflict_factor_stats_with_weights.csv", index=False)


def main() -> None:
    if not SURVEY_FILE.exists():
        raise FileNotFoundError(f"Missing required file: {SURVEY_FILE}")

    df = load_and_clean_survey(SURVEY_FILE)
    quality = dataset_quality_report(df)

    print("Dataset quality snapshot:")
    print(f"- Total rows: {quality.total_rows}")
    print(f"- Rows with usable severity labels: {quality.usable_rows}")
    print(f"- Likert non-null ratio (11 factor columns): {quality.likert_non_null_ratio:.4f}")

    model_out = train_random_forest(df)
    export_outputs(df, quality, model_out)

    print("\nModel training complete. Key metrics:")
    print(f"- Accuracy: {model_out['accuracy']:.4f}")
    print(f"- Weighted F1: {model_out['weighted_f1']:.4f}")
    print(f"\nSaved outputs to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
