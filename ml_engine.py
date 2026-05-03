"""
VetAI ML inference — shared by the Flask API and Streamlit UI.
Logic copied from the original app.py without modification to prediction behavior.
"""

from __future__ import annotations

import json
import os
import pickle
from typing import Any

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Order matches the Streamlit/UI checklist (used when mapping API feature arrays without a loaded bundle).
PREGNANCY_CHECKLIST_KEY_ORDER: tuple[str, ...] = (
    "cessation_of_estrus",
    "calmer_temperament",
    "increased_appetite",
    "weight_gain",
    "vulva_changes",
    "mucus_discharge",
    "uterine_asymmetry",
    "abdominal_swelling_right",
    "udder_development",
    "milk_vein_development",
    "fetal_movement",
    "pelvic_relaxation",
    "tail_ligament_softening",
    "isolation_behavior",
    "restlessness",
    "muscle_tremors",
    "colostrum_discharge",
    "rough_coat",
)


def _get_confidence_label(pct: float) -> str:
    if pct >= 85:
        return "High"
    if pct >= 65:
        return "Moderate"
    return "Low"


def rule_based_pregnancy_score(symptoms_dict: dict[str, int], days_since_breeding: int = 0):
    ces = symptoms_dict.get("cessation_of_estrus", 0)
    calm = symptoms_dict.get("calmer_temperament", 0)
    appetite = symptoms_dict.get("increased_appetite", 0)
    vulva = symptoms_dict.get("vulva_changes", 0)
    mucus = symptoms_dict.get("mucus_discharge", 0)
    uterine = symptoms_dict.get("uterine_asymmetry", 0)
    abdom = symptoms_dict.get("abdominal_swelling_right", 0)
    udder = symptoms_dict.get("udder_development", 0)
    milk_vein = symptoms_dict.get("milk_vein_development", 0)
    fetal = symptoms_dict.get("fetal_movement", 0)
    pelvic = symptoms_dict.get("pelvic_relaxation", 0)
    tail_lig = symptoms_dict.get("tail_ligament_softening", 0)
    isolation = symptoms_dict.get("isolation_behavior", 0)
    restless = symptoms_dict.get("restlessness", 0)
    colostrum = symptoms_dict.get("colostrum_discharge", 0)

    if pelvic and tail_lig and (restless or colostrum) and udder:
        return "Imminent Calving", 0.95
    if abdom and udder and (fetal or milk_vein) and ces:
        conf = 0.90 + (0.03 if pelvic else 0) + (0.02 if isolation else 0)
        return "Third Trimester (7-9 months)", min(0.96, conf)
    if ces and abdom and (fetal or isolation) and appetite:
        return "Second Trimester (3-6 months)", 0.88
    if ces and (uterine or (calm and appetite and vulva)):
        conf = 0.82 + (0.05 if uterine else 0) + (0.03 if vulva else 0)
        return "Mid Pregnancy (4-8 weeks)", min(0.90, conf)
    if ces and (vulva or mucus or calm):
        conf = 0.75 + (0.08 if days_since_breeding >= 21 else 0)
        return "Early Pregnancy (1-4 weeks)", min(0.88, conf)
    if not ces and not calm and not abdom:
        return "Not Pregnant", 0.80
    return None, 0.0


def _build_pregnancy_result(stage: str, confidence: float, kb: dict, days: int) -> dict[str, Any]:
    expected_calving = None
    if days > 0 and stage not in ["Not Pregnant"]:
        remaining_days = max(0, 283 - days)
        expected_calving = f"~{remaining_days} days from now"

    return {
        "stage": stage,
        "confidence": round(confidence * 100, 1),
        "confidence_level": _get_confidence_label(confidence * 100),
        "description": kb.get("description", ""),
        "recommendations": kb.get("recommendations", ""),
        "care": kb.get("care", ""),
        "confidence_note": kb.get("confidence_note", ""),
        "expected_calving_in": expected_calving,
        "days_since_breeding": days,
    }


def predict_disease(
    symptoms_dict: dict[str, int],
    disease_bundle: dict | None,
    knowledge_base: dict | None,
) -> dict[str, Any]:
    if disease_bundle is None:
        return {"error": "Disease model not found. On your computer, run: python train_models.py — then upload the files in the models/ folder to Hugging Face."}

    symptom_cols = disease_bundle["symptom_cols"]
    model = disease_bundle["model"]
    le = disease_bundle["label_encoder"]

    row = [
        symptoms_dict.get(
            col.lower().replace(" ", "_").replace("(", "").replace(")", ""),
            0,
        )
        for col in symptom_cols
    ]
    X = np.array([row])
    probas = model.predict_proba(X)[0]
    classes = le.classes_

    dk = (knowledge_base or {}).get("disease_knowledge", {})
    results = []
    for cls, prob in zip(classes, probas):
        kb = dk.get(cls, {})
        results.append(
            {
                "disease": cls,
                "confidence": round(float(prob) * 100, 1),
                "severity": kb.get("severity", "Unknown"),
                "urgency": kb.get("urgency", "Consult your veterinarian"),
                "description": kb.get("description", ""),
                "treatment": kb.get("treatment", ""),
                "prevention": kb.get("prevention", ""),
                "zoonotic": kb.get("zoonotic", False),
            }
        )

    results.sort(key=lambda x: x["confidence"], reverse=True)
    top = results[0]
    return {
        "top_prediction": top,
        "all_predictions": results[:5],
        "total_symptoms_entered": int(sum(symptoms_dict.values())),
        "confidence_level": _get_confidence_label(top["confidence"]),
    }


def predict_pregnancy(
    symptoms_dict: dict[str, int],
    days_since_breeding: int,
    pregnancy_bundle: dict | None,
    knowledge_base: dict | None,
) -> dict[str, Any]:
    rule_stage, rule_conf = rule_based_pregnancy_score(symptoms_dict, days_since_breeding)
    pk = (knowledge_base or {}).get("pregnancy_knowledge", {})

    if pregnancy_bundle is None:
        if rule_stage:
            kb = pk.get(rule_stage, {})
            return _build_pregnancy_result(rule_stage, rule_conf, kb, days_since_breeding)
        return {"error": "Pregnancy model not found. Run python train_models.py locally, then upload models/pregnancy_model.pkl to your Space."}

    feature_cols = pregnancy_bundle["feature_cols"]
    model = pregnancy_bundle["model"]
    le = pregnancy_bundle["label_encoder"]

    X_features = [symptoms_dict.get(col, 0) for col in feature_cols]
    X_features.append(days_since_breeding)
    X = np.array([X_features])

    probas = model.predict_proba(X)[0]
    ml_stage = le.classes_[int(np.argmax(probas))]
    ml_conf = float(np.max(probas))

    if rule_stage and rule_conf > 0.85:
        final_stage, final_conf, method = rule_stage, rule_conf, "Rule-based (expert patterns)"
    elif rule_stage and ml_stage == rule_stage:
        final_stage = ml_stage
        final_conf = min(0.97, (rule_conf + ml_conf) / 2 + 0.05)
        method = "Hybrid (rules + machine learning agree)"
    elif rule_stage:
        final_stage = rule_stage
        final_conf = rule_conf * 0.6 + ml_conf * 0.4
        method = "Mostly rule-based (with ML support)"
    else:
        final_stage, final_conf, method = ml_stage, ml_conf, "Machine learning"

    kb = pk.get(final_stage, {})
    result = _build_pregnancy_result(final_stage, final_conf, kb, days_since_breeding)
    result["method"] = method

    alternatives = []
    for cls, prob in zip(le.classes_, probas):
        if cls != final_stage:
            alternatives.append({"stage": cls, "probability": round(float(prob) * 100, 1)})
    alternatives.sort(key=lambda x: x["probability"], reverse=True)
    result["alternatives"] = alternatives[:3]
    return result


def load_ml_artifacts():
    disease_path = os.path.join(MODELS_DIR, "disease_model.pkl")
    pregnancy_path = os.path.join(MODELS_DIR, "pregnancy_model.pkl")
    kb_path = os.path.join(MODELS_DIR, "knowledge_base.json")

    disease_bundle = None
    pregnancy_bundle = None
    knowledge_base = None

    if os.path.exists(disease_path):
        with open(disease_path, "rb") as f:
            disease_bundle = pickle.load(f)
    if os.path.exists(pregnancy_path):
        with open(pregnancy_path, "rb") as f:
            pregnancy_bundle = pickle.load(f)
    if os.path.exists(kb_path):
        with open(kb_path, encoding="utf-8") as f:
            knowledge_base = json.load(f)

    return disease_bundle, pregnancy_bundle, knowledge_base


def normalized_disease_key(col: str) -> str:
    return col.lower().replace(" ", "_").replace("(", "").replace(")", "")


def features_array_to_disease_symptoms(feature_values: list, disease_bundle: dict) -> dict[str, int]:
    symptom_cols = disease_bundle["symptom_cols"]
    if len(feature_values) != len(symptom_cols):
        raise ValueError(
            f"disease features length {len(feature_values)} does not match model ({len(symptom_cols)} columns)"
        )
    return {
        normalized_disease_key(symptom_cols[i]): int(feature_values[i])
        for i in range(len(symptom_cols))
    }


def features_array_to_pregnancy_symptoms(feature_values: list, pregnancy_bundle: dict) -> dict[str, int]:
    feature_cols = pregnancy_bundle["feature_cols"]
    if len(feature_values) != len(feature_cols):
        raise ValueError(
            f"pregnancy features length {len(feature_values)} does not match model ({len(feature_cols)} columns)"
        )
    return {feature_cols[i]: int(feature_values[i]) for i in range(len(feature_cols))}


def pregnancy_api_features_to_symptoms(feature_values: list, pregnancy_bundle: dict | None) -> dict[str, int]:
    """Map a `features` array to symptom dicts; uses model column order when the bundle exists (ML path)."""
    if pregnancy_bundle is not None:
        return features_array_to_pregnancy_symptoms(feature_values, pregnancy_bundle)
    if len(feature_values) != len(PREGNANCY_CHECKLIST_KEY_ORDER):
        raise ValueError(
            f"pregnancy features length {len(feature_values)} does not match checklist ({len(PREGNANCY_CHECKLIST_KEY_ORDER)}); "
            "order matches the original VetAI pregnancy UI."
        )
    return {PREGNANCY_CHECKLIST_KEY_ORDER[i]: int(feature_values[i]) for i in range(len(feature_values))}
