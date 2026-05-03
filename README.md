---
title: VetAI — Cattle Health Assistant
emoji: 🐾
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.31.0
app_file: app.py
pinned: false
---

# VetAI — Cattle Health Intelligence (Streamlit + Hugging Face)

This repository contains **VetAI**, a decision-support demo for **cattle disease patterns** (classification from symptom checklists) and a **pregnancy staging helper**. The machine learning models are **not** trained on photos; they use the same symptom columns as `data/cattle_diseases.csv` and `data/pregnancy_stages.csv`.

**Disclaimer:** This is an educational, AI-assisted tool. It is **not** a medical diagnosis. Always consult a licensed veterinarian.

---

## What is in this project?

| Location | What it is |
|----------|------------|
| `app.py` | **Streamlit** app (used by Hugging Face Spaces and for local `streamlit run`). |
| `flask_app.py` | Original **Flask** API + server (optional; used if you run `run.bat` / `run.sh`). |
| `train_models.py` | Trains and writes `models/disease_model.pkl`, `models/pregnancy_model.pkl`, and refreshes `models/knowledge_base.json`. |
| `data/` | Sample CSVs used for training. |
| `models/` | Trained `.pkl` files and `knowledge_base.json` (create by training locally, then upload to your Space). |

**Model types**

- **Disease model:** multi-class classification (which cattle disease pattern best matches the checked symptoms).
- **Pregnancy model:** multi-class stage estimate (rule + random forest hybrid), plus optional `days_since_breeding`.

**Inputs the models expect**

- **Disease:** binary (0/1) flags for each column in `cattle_diseases.csv` except `disease` (for example `fever`, `cough`, `diarrhea`, …).
- **Pregnancy:** binary flags from `pregnancy_stages.csv` feature columns plus an integer `days_since_breeding`.

There is **no image / CNN model** in this repo. The Streamlit “Image analysis” tab explains this and lets you attach a photo only for your downloadable text report.

---

## Test the Streamlit app on your computer

### 1) Install Python

Install **Python 3.10 or 3.11** from [https://www.python.org/downloads/](https://www.python.org/downloads/). On Windows, enable **“Add Python to PATH”**.

### 2) Open a terminal in this folder

Example (Windows PowerShell):

```powershell
cd "C:\Users\YOUR_NAME\...\vetai"
```

### 3) Create a virtual environment (recommended)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

On Mac/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4) Install dependencies

```bash
pip install -r requirements.txt
```

### 5) Train the models (creates the `.pkl` files)

```bash
python train_models.py
```

You should see new files under `models/` (for example `disease_model.pkl` and `pregnancy_model.pkl`).

### 6) Run Streamlit

```bash
streamlit run app.py
```

Your browser should open at `http://localhost:8501`.

### Before you deploy — quick checklist

- [ ] `python train_models.py` completed without errors.
- [ ] `models/disease_model.pkl` and `models/pregnancy_model.pkl` exist.
- [ ] `models/knowledge_base.json` exists.
- [ ] `streamlit run app.py` works locally.

---

## Deploy to Hugging Face Spaces (step by step)

### A) Create a Hugging Face account

1. Open [https://huggingface.co/join](https://huggingface.co/join).
2. Sign up with email or Google/GitHub.
3. Confirm your email if asked.

### B) Create a new Space

1. Log in and go to [https://huggingface.co/new-space](https://huggingface.co/new-space).
2. **Space name:** pick a short name (for example `vetai-cattle-demo`).
3. **License:** choose what fits your project (many demos use Apache 2.0 or MIT).
4. **SDK:** select **Streamlit**.
5. **Hardware:** **CPU basic** is enough for this sklearn app.
6. Click **Create Space**.

### C) Upload your project files

Use the **Files** tab in the Space (or `git clone` + `git push` if you use Git).

**Minimum set for the app to run:**

- `app.py`
- `requirements.txt`
- `.streamlit/config.toml` (optional but recommended for theme)
- `README.md` (this file — the YAML block at the top helps Hugging Face detect Streamlit settings)
- **`models/disease_model.pkl`**
- **`models/pregnancy_model.pkl`**
- **`models/knowledge_base.json`**

**Strongly recommended (so others can retrain or audit):**

- `data/cattle_diseases.csv`
- `data/pregnancy_stages.csv`
- `train_models.py`

**Where do the model files go?**

Put them in a folder named **`models/`** at the **root** of the Space repository, next to `app.py`:

```text
your-space/
  app.py
  requirements.txt
  README.md
  .streamlit/
    config.toml
  models/
    disease_model.pkl
    pregnancy_model.pkl
    knowledge_base.json
```

### D) Wait for the build

The Space shows **Building** then **Running**. First build can take several minutes.

### E) How to know it succeeded

- Status badge shows **Running**.
- The app UI loads without a red error banner.
- Sidebar “Status” shows **Ready** for disease and pregnancy models.

### F) If something fails — logs

1. Open your Space.
2. Click **Logs** (or the “Logs” / “Build” output depending on UI version).
3. Read from the bottom upward for the first `Traceback` or `ModuleNotFoundError`.

**Common problems**

| Symptom | What it usually means | What to do |
|--------|------------------------|------------|
| `FileNotFoundError` for `.pkl` | Models not uploaded or wrong path | Confirm `models/` at repo root; re-run `train_models.py` locally and upload the files |
| `ModuleNotFoundError` | Missing dependency | Add the package to `requirements.txt` and save |
| Build OK but page blank | Streamlit crashed at runtime | Read **Logs**; fix the exception shown |
| Very slow first prediction | Cold start on small CPU | Wait a few seconds; normal on free hardware |

**Model fails to load**

- Re-train locally: `python train_models.py`.
- Confirm you did not upload empty or corrupted files.
- Make sure scikit-learn version is compatible (this repo pins versions in `requirements.txt`).

---

## Legacy Flask UI (optional)

If you prefer the original HTML interface on port 5000:

```bash
pip install -r requirements.txt
python train_models.py
python -c "from flask_app import app, load_models; load_models(); app.run(debug=False, host='0.0.0.0', port=5000)"
```

Or run `run.bat` (Windows) / `run.sh` (Mac/Linux) after editing paths if needed.

---

## How the AI side fits together (short)

- **Training:** `train_models.py` reads `data/*.csv`, augments rows, trains sklearn models, writes pickles under `models/`.
- **Streamlit inference:** `app.py` loads pickles once (`@st.cache_resource`), builds feature vectors from your checkboxes, calls `predict_proba`, and shows the top classes with plain-language text from `knowledge_base.json`.

---

## License and safety

Use responsibly. For real animals, diagnosis and treatment decisions belong with a qualified veterinarian.
