"""
VetAI — Streamlit entry for Hugging Face Spaces and `streamlit run app.py`.
Delegates UI to backend/streamlit_ui.py; ML lives in backend/ml_engine.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

_backend = Path(__file__).resolve().parent / "backend"
sys.path.insert(0, str(_backend))

from streamlit_ui import main

main()
