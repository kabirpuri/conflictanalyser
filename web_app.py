#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

# Make vendored dependencies importable.
VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

from flask import Flask, abort, redirect, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "web_runs"
ALLOWED_EXTENSIONS = {".csv"}

APP_TITLE = "Construction Conflict AI Analyzer"
DEFAULT_SURVEY_FILENAME = "Cleaned_Construction_Conflict_Survey_Data.csv"
DEFAULT_STATS_FILENAME = "Conflict_Factor_Statistics_Final.csv"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB


def is_allowed_csv(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def run_analysis_for_job(job_dir: Path) -> None:
    env = os.environ.copy()
    # Ensure subprocesses use local vendored dependencies.
    env["PYTHONPATH"] = str(VENDOR_DIR)

    subprocess.run(
        ["python3", str(BASE_DIR / "conflict_model_pipeline.py")],
        cwd=str(job_dir),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    subprocess.run(
        ["python3", str(BASE_DIR / "run_experiments_and_plots.py")],
        cwd=str(job_dir),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def list_plot_files(job_dir: Path) -> list[str]:
    plots_dir = job_dir / "outputs" / "plots"
    if not plots_dir.exists():
        return []
    return sorted([p.name for p in plots_dir.glob("*.png")])


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _load_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except OSError:
        return []
    return rows


def _fmt_score(value: object) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "N/A"


def collect_result_context(job_dir: Path) -> dict:
    outputs_dir = job_dir / "outputs"
    metrics = _load_json(outputs_dir / "model_metrics.json")
    summary = _load_json(outputs_dir / "experiments_summary.json")
    tests = _load_csv_rows(outputs_dir / "model_test_results_cv.csv")
    top_features = _load_csv_rows(outputs_dir / "rf_feature_importance_top30_experiments.csv")[:10]

    best_name = "N/A"
    best_f1 = "N/A"
    best_acc = "N/A"
    if summary.get("best_by_f1_weighted_cv"):
        best = summary["best_by_f1_weighted_cv"]
        best_name = str(best.get("test_name", "N/A"))
        best_f1 = _fmt_score(best.get("f1_weighted_mean"))
        best_acc = _fmt_score(best.get("accuracy_mean"))

    holdout = summary.get("rf_holdout_metrics", {})
    holdout_acc = _fmt_score(holdout.get("holdout_accuracy"))
    holdout_f1 = _fmt_score(holdout.get("holdout_weighted_f1"))

    return {
        "metrics": metrics,
        "summary": summary,
        "tests": tests,
        "top_features": top_features,
        "best_name": best_name,
        "best_f1": best_f1,
        "best_acc": best_acc,
        "holdout_acc": holdout_acc,
        "holdout_f1": holdout_f1,
    }


def write_markdown_report(job_dir: Path, context: dict) -> Path:
    outputs_dir = job_dir / "outputs"
    metrics = context["metrics"]
    tests = context["tests"]
    top_features = context["top_features"]

    report_lines = [
        "# Construction Conflict Analysis Report",
        "",
        "## 1. Study Goal",
        "This report summarizes an AI-assisted severity prediction workflow for construction conflict data.",
        "",
        "## 2. Theory and Methodology",
        "- Conflict severity is treated as a 3-class problem: Low, Medium, High.",
        "- Features include demographics and NLP features from qualitative responses.",
        "- Text is converted using TF-IDF (unigrams and bigrams).",
        "- Models are evaluated with repeated stratified cross-validation.",
        "- Final diagnostics include confusion matrix and feature importance.",
        "",
        "## 3. Data Quality Snapshot",
        f"- Total rows: {metrics.get('total_rows', 'N/A')}",
        f"- Usable rows: {metrics.get('usable_rows', 'N/A')}",
        f"- Likert non-null ratio: {metrics.get('likert_non_null_ratio', 'N/A')}",
        "",
        "## 4. Model Performance",
        f"- Best CV model (weighted F1): {context['best_name']}",
        f"- Best CV weighted F1: {context['best_f1']}",
        f"- Best CV accuracy: {context['best_acc']}",
        f"- Random Forest holdout accuracy: {context['holdout_acc']}",
        f"- Random Forest holdout weighted F1: {context['holdout_f1']}",
        "",
        "### Cross-validation test results",
        "",
        "| Test | Accuracy Mean | Accuracy Std | Weighted F1 Mean | Weighted F1 Std |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in tests:
        report_lines.append(
            f"| {row.get('test_name', '')} | {row.get('accuracy_mean', '')} | {row.get('accuracy_std', '')} | "
            f"{row.get('f1_weighted_mean', '')} | {row.get('f1_weighted_std', '')} |"
        )

    report_lines.extend(
        [
            "",
            "## 5. Top Predictive Features",
            "",
            "| Rank | Feature | Importance |",
            "|---:|---|---:|",
        ]
    )
    for idx, row in enumerate(top_features, start=1):
        report_lines.append(f"| {idx} | {row.get('feature', '')} | {row.get('importance', '')} |")

    report_lines.extend(
        [
            "",
            "## 6. Plots Included",
            "- Class distribution",
            "- Model comparison",
            "- Confusion matrix (Random Forest)",
            "- Top feature importances",
            "- Conflict factor ranking (if stats CSV was provided)",
            "- Conflict factor normalized weights (if stats CSV was provided)",
            "",
            "## 7. Interpretation and Limitations",
            "- Results are exploratory because the dataset is small.",
            "- Severity predictions are useful for early warning, not final legal decisions.",
            "- If respondent-level Likert factors are complete, the same pipeline can be extended to compute stronger factor-level risk scoring.",
        ]
    )

    report_path = outputs_dir / "analysis_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return report_path


@app.get("/")
def index():
    return render_template("index.html", app_title=APP_TITLE)


@app.post("/analyze")
def analyze():
    survey_file = request.files.get("survey_csv")
    stats_file = request.files.get("stats_csv")

    if survey_file is None or survey_file.filename == "":
        return "Please upload the survey CSV file.", 400
    if not is_allowed_csv(survey_file.filename):
        return "Survey file must be a CSV.", 400

    if stats_file and stats_file.filename and not is_allowed_csv(stats_file.filename):
        return "Statistics file must be a CSV.", 400

    job_id = uuid.uuid4().hex[:12]
    job_dir = RUNS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    survey_name = secure_filename(survey_file.filename) or DEFAULT_SURVEY_FILENAME
    saved_survey = job_dir / survey_name
    survey_file.save(saved_survey)
    # Pipeline expects this fixed file name.
    if survey_name != DEFAULT_SURVEY_FILENAME:
        shutil.copy2(saved_survey, job_dir / DEFAULT_SURVEY_FILENAME)

    if stats_file and stats_file.filename:
        stats_name = secure_filename(stats_file.filename) or DEFAULT_STATS_FILENAME
        saved_stats = job_dir / stats_name
        stats_file.save(saved_stats)
        if stats_name != DEFAULT_STATS_FILENAME:
            shutil.copy2(saved_stats, job_dir / DEFAULT_STATS_FILENAME)
    else:
        default_stats = BASE_DIR / DEFAULT_STATS_FILENAME
        if default_stats.exists():
            shutil.copy2(default_stats, job_dir / DEFAULT_STATS_FILENAME)

    try:
        run_analysis_for_job(job_dir)
    except subprocess.CalledProcessError as exc:
        error_log = (
            f"STDOUT:\n{exc.stdout}\n\nSTDERR:\n{exc.stderr}\n"
            if exc.stdout or exc.stderr
            else str(exc)
        )
        (job_dir / "error.log").write_text(error_log, encoding="utf-8")
        return render_template(
            "result.html",
            app_title=APP_TITLE,
            job_id=job_id,
            success=False,
            plots=[],
            message="Analysis failed. Check error log.",
        )

    zip_path = shutil.make_archive(str(job_dir / "analysis_outputs"), "zip", root_dir=job_dir / "outputs")

    result_context = collect_result_context(job_dir)
    write_markdown_report(job_dir, result_context)

    return render_template(
        "result.html",
        app_title=APP_TITLE,
        job_id=job_id,
        success=True,
        plots=list_plot_files(job_dir),
        zip_name=Path(zip_path).name,
        result_context=result_context,
    )


@app.get("/job/<job_id>/plot/<plot_name>")
def serve_plot(job_id: str, plot_name: str):
    job_dir = RUNS_DIR / job_id
    plot_path = job_dir / "outputs" / "plots" / secure_filename(plot_name)
    if not plot_path.exists():
        abort(404)
    return send_file(plot_path)


@app.get("/job/<job_id>/download")
def download_zip(job_id: str):
    job_dir = RUNS_DIR / job_id
    zip_path = job_dir / "analysis_outputs.zip"
    if not zip_path.exists():
        abort(404)
    return send_file(zip_path, as_attachment=True, download_name=f"analysis_outputs_{job_id}.zip")


@app.get("/job/<job_id>/report")
def view_report(job_id: str):
    job_dir = RUNS_DIR / job_id
    if not job_dir.exists():
        abort(404)
    context = collect_result_context(job_dir)
    plots = list_plot_files(job_dir)
    return render_template(
        "report.html",
        app_title=APP_TITLE,
        job_id=job_id,
        result_context=context,
        plots=plots,
    )


@app.get("/job/<job_id>/report/download")
def download_report(job_id: str):
    job_dir = RUNS_DIR / job_id
    report_path = job_dir / "outputs" / "analysis_report.md"
    if not report_path.exists():
        abort(404)
    return send_file(report_path, as_attachment=True, download_name=f"analysis_report_{job_id}.md")


@app.get("/job/<job_id>/error")
def download_error(job_id: str):
    job_dir = RUNS_DIR / job_id
    log_path = job_dir / "error.log"
    if not log_path.exists():
        abort(404)
    return send_file(log_path, as_attachment=True, download_name=f"analysis_error_{job_id}.log")


@app.get("/new")
def new_job():
    return redirect(url_for("index"))


if __name__ == "__main__":
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    app.run(host=host, port=port, debug=False)
