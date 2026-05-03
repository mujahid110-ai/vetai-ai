"""
VetAI Lite: single Flask service for easiest free deployment.
Deploy this root directly on Render/Railway with Procfile + runtime.txt.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from ml_engine import (  # noqa: E402
    PREGNANCY_CHECKLIST_KEY_ORDER,
    features_array_to_disease_symptoms,
    load_ml_artifacts,
    predict_disease,
    predict_pregnancy,
    pregnancy_api_features_to_symptoms,
)

app = Flask(__name__)

_disease_bundle = None
_pregnancy_bundle = None
_knowledge_base = None


def _init_artifacts() -> None:
    global _disease_bundle, _pregnancy_bundle, _knowledge_base
    _disease_bundle, _pregnancy_bundle, _knowledge_base = load_ml_artifacts()


@app.after_request
def _cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = os.environ.get("CORS_ORIGIN", "*")
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return resp


@app.route("/", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "service": "vetai-lite",
            "disease_model": _disease_bundle is not None,
            "pregnancy_model": _pregnancy_bundle is not None,
            "knowledge_base": _knowledge_base is not None,
            "usage": "POST /predict with {task: disease|pregnancy, features: [...], days_since_breeding?: int}",
        }
    )


@app.route("/schema", methods=["GET"])
def schema():
    return jsonify(
        {
            "disease_symptom_columns": _disease_bundle["symptom_cols"] if _disease_bundle else None,
            "pregnancy_feature_columns": _pregnancy_bundle["feature_cols"] if _pregnancy_bundle else None,
            "pregnancy_checklist_fallback_order": list(PREGNANCY_CHECKLIST_KEY_ORDER),
        }
    )


@app.route("/predict", methods=["OPTIONS"])
def predict_options():
    return ("", 204)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    task = (data.get("task") or "").strip().lower()
    features = data.get("features")

    if task not in ("disease", "pregnancy"):
        return jsonify({"error": 'Invalid "task". Use "disease" or "pregnancy".'}), 400
    if not isinstance(features, list):
        return jsonify({"error": '"features" must be a JSON array.'}), 400

    try:
        if task == "disease":
            if _disease_bundle is None:
                out = predict_disease({}, None, _knowledge_base)
            else:
                symptoms = features_array_to_disease_symptoms(features, _disease_bundle)
                out = predict_disease(symptoms, _disease_bundle, _knowledge_base)
        else:
            days = int(data.get("days_since_breeding", 0))
            symptoms = pregnancy_api_features_to_symptoms(features, _pregnancy_bundle)
            out = predict_pregnancy(symptoms, days, _pregnancy_bundle, _knowledge_base)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    if isinstance(out, dict) and out.get("error"):
        return jsonify({"error": out["error"]}), 503
    return jsonify({"prediction": out})


_init_artifacts()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)