---
title: VetAI
emoji: ??
colorFrom: green
colorTo: blue
sdk: streamlit
app_file: app.py
pinned: false
---

# VetAI (Hugging Face Streamlit)

Simple Streamlit deployment for Hugging Face Spaces.

## Required files
- `app.py`
- `ml_engine.py`
- `models/` (contains `.pkl` and `knowledge_base.json`)
- `requirements.txt`

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```