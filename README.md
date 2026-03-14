# Construction Conflict Modeling Pipeline

This repository contains a reproducible pipeline to model construction conflict severity from the available CSV files:

- `Cleaned_Construction_Conflict_Survey_Data.csv`
- `Conflict_Factor_Statistics_Final.csv`

## What the pipeline does

`conflict_model_pipeline.py` performs:

1. Data cleaning and normalization
2. Conflict severity label mapping to numeric classes (`Low=1`, `Medium=2`, `High=3`)
3. Feature engineering using:
   - Categorical one-hot features (`Role`, `Experience`, `Project_Sector`)
   - TF-IDF text features (`Trigger_Event`, `Conflict_Progression`, `Hidden_Factors`, `Additional_Comments`)
4. Random Forest classification
5. Export of model metrics and top feature importances

## Run

```bash
python3 conflict_model_pipeline.py
```

If dependencies are missing in your system Python, install them locally in this repo:

```bash
mkdir -p vendor
pip3 install --target ./vendor scikit-learn scipy
```

## Output files

The script writes outputs to `outputs/`:

- `survey_cleaned_for_model.csv`
- `model_metrics.json`
- `classification_report.txt`
- `top_feature_importance.csv`
- `conflict_factor_stats_with_weights.csv` (if factor statistics file exists)

## Important data note

In the current `Cleaned_Construction_Conflict_Survey_Data.csv`, all 11 Likert factor columns are empty.  
So this pipeline uses demographics + text responses for severity modeling.  
If you later restore row-level Likert values, the script can be extended to include factor-level predictive modeling and CRI at respondent level.

## Simple website for anyone to use

A simple upload-and-run web interface is included in `web_app.py`.

### Install local dependencies

```bash
mkdir -p vendor
pip3 install --target ./vendor scikit-learn scipy matplotlib flask
```

### Standard setup (recommended for GitHub users)

```bash
pip3 install -r requirements.txt
```

### Run website

```bash
python3 web_app.py
```

If port 8000 is already in use:

```bash
PORT=8001 python3 web_app.py
```

Open `http://localhost:8000` and:
1. Upload survey CSV (required)
2. Upload factor-statistics CSV (optional)
3. Click **Run Analysis**
4. Download the ZIP with plots/tables/metrics
5. Open the built-in full report page (theory + methods + benchmark + plots)

Each run is isolated in `web_runs/<job_id>/` so multiple users can run separate analyses.

## Host on `conflictanalyser.github.io`

Important: GitHub Pages cannot run Python/Flask directly.  
Use this architecture:

- `conflictanalyser.github.io` -> static frontend (`index.html` in this repo)
- Render free web service -> Flask backend (`web_app.py`)

### 1) Push this repo to GitHub user-site repository

Create a repository named exactly:

`conflictanalyser.github.io`

Then push:

```bash
git remote add origin https://github.com/conflictanalyser/conflictanalyser.github.io.git
git push -u origin main
```

### 2) Enable GitHub Pages

In repository settings:

- Pages -> Build and deployment -> Deploy from branch
- Branch: `main`
- Folder: `/ (root)`

Your frontend URL becomes:

`https://conflictanalyser.github.io`

### 3) Deploy backend on Render (free)

- Create a new Web Service from this same GitHub repo
- Render will detect `render.yaml` (or use below commands)
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn --bind 0.0.0.0:$PORT web_app:app`

After deploy, you get a backend URL like:

`https://conflict-analyser-backend.onrender.com`

### 4) Connect frontend to backend

In `index.html`, update:

```js
const BACKEND_URL = "https://your-render-url.onrender.com";
```

Commit and push again.  
Now `https://conflictanalyser.github.io` will show your branded frontend and embed/open the live analyzer app.
