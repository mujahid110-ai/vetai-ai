"""
Production Flask API for VetAI ML inference (Render / gunicorn).
"""

from __future__ import annotations

import os

from flask import Flask, jsonify, request

from ml_engine import (
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


def _init_artifacts():
    global _disease_bundle, _pregnancy_bundle, _knowledge_base
    _disease_bundle, _pregnancy_bundle, _knowledge_base = load_ml_artifacts()


def _add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = os.environ.get("CORS_ORIGIN", "*")
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    resp.headers["Access-Control-Max-Age"] = "86400"
    return resp


@app.after_request
def cors_after(resp):
    return _add_cors(resp)


@app.route("/", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "disease_model": _disease_bundle is not None,
            "pregnancy_model": _pregnancy_bundle is not None,
            "knowledge_base": _knowledge_base is not None,
        }
    )


@app.route("/schema", methods=["GET"])
def schema():
    """Column order for building `features` arrays (same order as training)."""
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
    """
    JSON body:
      { "task": "disease" | "pregnancy", "features": [...], "days_since_breeding": int (pregnancy only) }
    Response:
      { "prediction": <model output dict> } or { "error": "..." }
    """
    data = request.get_json(silent=True) or {}
    task = (data.get("task") or "").strip().lower()
    features = data.get("features")

    if task not in ("disease", "pregnancy"):
        return jsonify({"error": 'Missing or invalid "task"; use "disease" or "pregnancy".'}), 400
    if not isinstance(features, list):
        return jsonify({"error": '"features" must be a JSON array of numbers (0/1) in training column order.'}), 400

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
