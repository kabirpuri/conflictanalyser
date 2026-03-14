# Construction Conflict Analyzer (Beginner Guide)

This project gives you a simple website to run conflict analysis on CSV files.
You do **not** need GitHub or coding experience to use it locally.

---

## What this app does

- Upload your survey CSV
- Run multiple model tests automatically
- Generate plots for report/thesis
- Show a full theory + methodology + result report page
- Let you download everything as one ZIP

---

## Before you start (one-time setup)

You only need a terminal and internet for package install.

### 1) Open terminal in this project folder

If your folder is:

`/home/kabirpuri/Downloads/bansi_project`

run:

```bash
cd /home/kabirpuri/Downloads/bansi_project
```

### 2) Install required packages (one-time)

Copy-paste this exactly:

```bash
mkdir -p vendor
pip3 install --target ./vendor -r requirements.txt
```

If `pip3` is missing:

```bash
sudo apt update
sudo apt install -y python3-pip
```

Then run the install command again.

---

## Run the website (every time)

From the same project folder:

```bash
python3 web_app.py
```

If you get "port already in use", run:

```bash
PORT=8020 python3 web_app.py
```

Now open your browser and go to:

- `http://127.0.0.1:8000`  
or, if you used custom port:
- `http://127.0.0.1:8020`

---

## How to use the website

1. Open the page in browser.
2. In **Survey CSV (required)**, upload your survey file.
3. In **Factor Statistics CSV (optional)**, upload factor stats file.
4. Click **Run Full Analysis**.
5. Wait until results page appears.
6. Use:
   - **Download All Outputs (ZIP)**
   - **Open Full Report**
   - **Download Report (Markdown)**

---

## Where output files are saved

Each run is saved in a separate folder:

- `web_runs/<job_id>/outputs/`

Important files inside:

- `model_test_results_cv.csv`
- `experiments_summary.json`
- `analysis_report.md`
- `plots/*.png`

---

## If you just want to run scripts (without website)

From project folder:

```bash
python3 conflict_model_pipeline.py
python3 run_experiments_and_plots.py
```

This writes outputs to:

- `outputs/`

---

## Common problems and quick fixes

### Problem: `python3: command not found`

Install Python:

```bash
sudo apt update
sudo apt install -y python3
```

### Problem: package import errors (`flask`, `sklearn`, etc.)

Run again:

```bash
mkdir -p vendor
pip3 install --target ./vendor -r requirements.txt
```

### Problem: website does not open

1. Check terminal is still running.
2. Use a different port:

```bash
PORT=8030 python3 web_app.py
```

3. Open `http://127.0.0.1:8030`.

### Problem: stop the running website

In the terminal where app is running, press:

`Ctrl + C`

---

## Data note

Current included file `Cleaned_Construction_Conflict_Survey_Data.csv` has empty row-level Likert factor columns.
So the model currently learns mainly from demographics + text fields for severity prediction.
