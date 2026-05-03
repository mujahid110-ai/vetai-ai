"""
VetAI — Streamlit UI (Hugging Face Spaces and local). ML lives in ml_engine.py.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import streamlit as st
from PIL import Image

from ml_engine import load_ml_artifacts, predict_disease, predict_pregnancy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

DISEASE_SYMPTOM_LABELS: dict[str, str] = {
    "fever": "Fever or high temperature",
    "cough": "Coughing",
    "nasal_discharge": "Runny nose / nasal discharge",
    "diarrhea": "Diarrhea",
    "lethargy": "Lethargy (very tired / weak)",
    "loss_of_appetite": "Loss of appetite",
    "weight_loss": "Weight loss",
    "lameness": "Lameness (limping)",
    "swelling": "Swelling on the body or legs",
    "skin_lesions": "Blisters, sores, or skin lesions",
    "breathing_difficulty": "Trouble breathing",
    "eye_discharge": "Eye discharge or squinting",
    "salivation": "Drooling / extra saliva",
    "bloat": "Bloat (very round belly, uncomfortable)",
    "pale_mucous_membranes": "Pale gums or inner eyelids",
    "milk_decrease": "Drop in milk production",
    "reproductive_issues": "Reproductive problems (abortion, discharge, etc.)",
    "nervous_signs": "Nervous signs (trembling, circling, unusual behavior)",
    "rough_coat": "Rough or dull coat",
    "dehydration": "Dehydration (sunken eyes, dry mouth)",
    "mucus_discharge": "Mucus discharge",
    "restlessness": "Restlessness",
    "isolation": "Keeping away from the herd",
    "udder_swelling": "Udder swelling or hardness",
    "muscle_tremors": "Muscle tremors or shaking",
    "foul_discharge": "Foul-smelling discharge",
    "recumbency": "Down and cannot stand",
    "sweet_breath": "Sweet or fruity breath smell",
    "staggering": "Staggering or wobbly walking",
}

PREGNANCY_FEATURE_LABELS: dict[str, str] = {
    "cessation_of_estrus": "Heat cycles have stopped (cessation of estrus)",
    "calmer_temperament": "Calmer temperament than usual",
    "increased_appetite": "Increased appetite",
    "weight_gain": "Gradual weight gain",
    "vulva_changes": "Vulva looks fuller or softer",
    "mucus_discharge": "Clear or stringy mucus from the vulva",
    "uterine_asymmetry": "One uterine horn feels larger (needs exam)",
    "abdominal_swelling_right": "Right side of belly looks fuller (rumen vs. pregnancy)",
    "udder_development": "Udder is developing or filling",
    "milk_vein_development": "Milk veins more visible",
    "fetal_movement": "Possible fetal movement felt on the flank",
    "pelvic_relaxation": "Pelvis feels wider / ligaments looser",
    "tail_ligament_softening": "Tail head looks raised; ligaments feel softer",
    "isolation_behavior": "Keeps away from the group",
    "restlessness": "Restless before calving",
    "muscle_tremors": "Muscle tremors (also seen with metabolic disease)",
    "colostrum_discharge": "Thick yellow fluid (colostrum) at teats",
    "rough_coat": "Rough coat (non-specific)",
}

DURATION_OPTIONS = [
    "1 day",
    "2–3 days",
    "About 1 week",
    "More than 1 week",
]

ANIMAL_OPTIONS = ["Cattle (this app is built for cattle)", "Dog", "Cat", "Bird", "Horse", "Other"]

DEMO_DISEASE_SYMPTOMS = ["fever", "cough", "nasal_discharge", "lethargy", "loss_of_appetite", "breathing_difficulty"]
DEMO_PREGNANCY_SIGNS = ["cessation_of_estrus", "abdominal_swelling_right", "udder_development", "fetal_movement"]


@st.cache_resource
def load_artifacts():
    return load_ml_artifacts()


def severity_tier(severity_text: str) -> str:
    s = (severity_text or "").lower()
    if any(k in s for k in ("critical", "immediate", "fatal", "notifiable", "anthrax", "severe")):
        return "high"
    if any(k in s for k in ("high", "moderate", "urgent")):
        return "medium"
    return "low"


def render_disclaimer():
    st.caption(
        "This is an AI-assisted educational tool. It is not a medical diagnosis. "
        "Always consult a licensed veterinarian for examination, testing, and treatment."
    )


def add_history(entry: dict[str, Any]) -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.insert(0, entry)
    st.session_state.history = st.session_state.history[:5]


def build_report_text(latest: dict[str, Any] | None) -> str:
    lines = [
        "VetAI — Visit summary (for your veterinarian)",
        f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC",
        "",
    ]
    if latest:
        lines.append(json.dumps(latest, indent=2, ensure_ascii=False))
    else:
        lines.append("No diagnosis stored in this session yet.")
    lines.extend(["", "Disclaimer: AI-assisted only. Not a substitute for professional veterinary care."])
    return "\n".join(lines)


def inject_style():
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px; }
        div[data-testid="stTabs"] button { font-weight: 600; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="VetAI — Veterinary Assistant",
        page_icon="🐾",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_style()

    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_image" not in st.session_state:
        st.session_state.last_image = None
    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    disease_bundle, pregnancy_bundle, knowledge_base = load_artifacts()

    with st.sidebar:
        st.header("Welcome")
        st.markdown(
            "VetAI helps you organize **cattle health clues** using a checklist and a trained model. "
            "It also includes a **pregnancy staging helper** for cows."
        )
        st.divider()
        st.subheader("How to use")
        st.markdown(
            "1. Open **Symptom Checker** and pick the signs you see.\n"
            "2. Tap **Get diagnosis** to see possible conditions ranked by probability.\n"
            "3. Use **Pregnancy** inside the same tab if you are tracking breeding.\n"
            "4. Review **Results History** for your last few checks this session."
        )
        st.divider()
        st.subheader("About")
        st.markdown(
            "This project combines machine learning with plain-language explanations. "
            "Trained labels come from cattle examples; it does **not** read X-rays or photos."
        )
        st.divider()
        st.caption("Status")
        st.write(
            {
                "Disease model": "Ready" if disease_bundle else "Missing file",
                "Pregnancy model": "Ready" if pregnancy_bundle else "Missing file",
                "Knowledge base": "Ready" if knowledge_base else "Missing file",
            }
        )
        sample_csv = os.path.join(DATA_DIR, "cattle_diseases.csv")
        if os.path.isfile(sample_csv):
            st.caption("Sample data folder detected — you can train models locally with train_models.py.")

    st.title("🐾 VetAI — Veterinary Diagnosis Assistant")
    st.markdown("**Cattle-focused** checklist assistant with gentle guidance — keep your vet in the loop.")
    render_disclaimer()

    tab_img, tab_sym, tab_hist = st.tabs(["Image analysis", "Symptom checker", "Results history"])

    with tab_img:
        st.info(
            "**Important:** This version of VetAI does **not** use photos for machine learning. "
            "The models were trained on **symptom checklists** (zeros and ones), not pictures. "
            "You can still attach a photo to your downloaded report for your veterinarian."
        )
        up = st.file_uploader(
            "Upload a photo (JPG, PNG, JPEG)",
            type=["jpg", "jpeg", "png"],
            help="Optional. Useful as a visual note for your vet. Not analyzed by the AI.",
        )
        if up is not None:
            try:
                image = Image.open(up).convert("RGB")
                st.image(image, caption="Your photo (for your records)", use_container_width=True)
                st.session_state.last_image = {"name": up.name, "pil": image}
            except Exception as e:
                st.error(f"We could not open that image. Try another file. ({e})")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Save photo to my report", help="Stores this image name in the session for the text report."):
                if st.session_state.last_image:
                    st.success("Saved. When you download a report, we will mention that a photo was attached in this session.")
                else:
                    st.warning("Upload a photo first.")
        with col_b:
            if st.button("Clear photo from session"):
                st.session_state.last_image = None
                st.rerun()

        st.markdown("#### What vets often look for in photos (general education)")
        st.markdown(
            "- Posture: is the animal standing normally, or stretched/stiff?\n"
            "- Eyes and nose: discharge, swelling, cloudiness?\n"
            "- Skin: swellings, sores, hair loss?\n"
            "- Belly: very tight ‘pear shape’ can be serious in cattle (bloat) — urgent vet call."
        )

    with tab_sym:
        animal = st.selectbox(
            "Animal type",
            ANIMAL_OPTIONS,
            help="The AI was trained for cattle. Other species are not supported by the model.",
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            age_years = st.number_input("Age (years)", min_value=0.0, max_value=40.0, value=3.0, step=0.5)
        with c2:
            weight_val = st.number_input("Weight", min_value=0.0, max_value=2000.0, value=450.0, step=1.0)
        with c3:
            wunit = st.selectbox("Weight unit", ["kg", "lbs"])

        weight_kg = weight_val * 0.453592 if wunit == "lbs" else weight_val

        duration = st.selectbox("How long have signs been present?", DURATION_OPTIONS)
        notes = st.text_area("Anything else we should know?", placeholder="E.g., recent diet change, new pasture, transport…")

        inner = st.tabs(["Illness checklist (cattle)", "Pregnancy helper (cattle)"])

        with inner[0]:
            st.markdown("Check every sign that fits. When unsure, leave it unchecked.")
            ill_key = "illness_multiselect"
            if ill_key not in st.session_state:
                st.session_state[ill_key] = []
            if st.button("Load example (demo)", key="demo_dis"):
                st.session_state[ill_key] = DEMO_DISEASE_SYMPTOMS.copy()
                st.rerun()
            picked = st.multiselect(
                "Symptoms and signs",
                options=list(DISEASE_SYMPTOM_LABELS.keys()),
                format_func=lambda k: DISEASE_SYMPTOM_LABELS[k],
                key=ill_key,
                help="These options match the training data for cattle diseases.",
            )

            symptoms_dict = {k: (1 if k in picked else 0) for k in DISEASE_SYMPTOM_LABELS.keys()}

            if animal != ANIMAL_OPTIONS[0]:
                st.warning(
                    "You selected a non-cattle animal. The model was trained on cattle patterns only. "
                    "Treat any result as **not reliable** for this species — call your veterinarian."
                )

            if st.button("Get diagnosis", type="primary", key="btn_dis"):
                ctx = {
                    "animal": animal,
                    "age_years": age_years,
                    "weight_kg": round(weight_kg, 2),
                    "duration": duration,
                    "notes": notes,
                    "symptoms": picked,
                }
                with st.spinner("Thinking through possibilities…"):
                    try:
                        out = predict_disease(symptoms_dict, disease_bundle, knowledge_base)
                    except Exception as e:
                        out = {"error": f"Something went wrong during prediction: {e}"}

                if "error" in out:
                    st.error(out["error"])
                else:
                    top = out["top_prediction"]
                    tier = severity_tier(top.get("severity", ""))
                    conf_frac = min(1.0, max(0.0, float(top["confidence"]) / 100.0))

                    st.subheader("Most likely pattern (not a final diagnosis)")
                    st.metric("Top possibility", top["disease"])
                    st.progress(conf_frac)
                    st.caption(f"Model confidence: {top['confidence']}%")

                    if tier == "high":
                        sev = "🔴 High concern — treat as urgent until a vet examines the animal"
                        st.error(sev)
                    elif tier == "medium":
                        sev = "🟡 Medium concern — contact your vet soon"
                        st.warning(sev)
                    else:
                        sev = "🟢 Lower urgency — still worth a vet visit if signs worsen"
                        st.success(sev)

                    st.markdown(f"**Plain summary:** {top.get('description') or 'No short description available.'}")
                    st.markdown(f"**What owners often discuss with their vet:** {top.get('treatment') or '—'}")
                    if top.get("zoonotic"):
                        st.warning("This condition may sometimes affect people (zoonotic). Use gloves and good hygiene; ask your vet.")

                    with st.expander("Other possibilities (top 5)"):
                        for row in out.get("all_predictions", []):
                            st.write(f"- **{row['disease']}** — {row['confidence']}% — severity note: {row['severity']}")

                    rec_actions = (
                        f"1) Call your veterinarian with this summary.\n"
                        f"2) Mention how long signs lasted ({duration}).\n"
                        f"3) Share weight (~{weight_kg:.1f} kg) and age ({age_years} yr).\n"
                        f"4) If breathing is hard, the belly looks very tight, or the animal is down: seek urgent care."
                    )
                    st.markdown("#### Suggested next steps")
                    st.markdown(rec_actions)

                    entry = {
                        "type": "disease",
                        "time_utc": datetime.utcnow().isoformat(),
                        "context": ctx,
                        "result": out,
                    }
                    st.session_state.last_result = entry
                    add_history(entry)

        with inner[1]:
            st.caption("Best for cows and heifers where breeding dates may be known.")
            days_sb = st.number_input(
                "Days since breeding (0 if unknown)",
                min_value=0,
                max_value=400,
                value=0,
                help="Approximate days since mating or insemination. If unknown, enter 0.",
            )
            preg_key = "pregnancy_multiselect"
            if preg_key not in st.session_state:
                st.session_state[preg_key] = []
            if st.button("Load example (demo)", key="demo_preg"):
                st.session_state[preg_key] = DEMO_PREGNANCY_SIGNS.copy()
                st.rerun()
            ppicked = st.multiselect(
                "Pregnancy-related signs",
                options=list(PREGNANCY_FEATURE_LABELS.keys()),
                format_func=lambda k: PREGNANCY_FEATURE_LABELS[k],
                key=preg_key,
            )

            preg_sym = {k: (1 if k in ppicked else 0) for k in PREGNANCY_FEATURE_LABELS.keys()}

            if st.button("Estimate pregnancy stage", type="primary", key="btn_preg"):
                ctx = {
                    "animal": animal,
                    "days_since_breeding": int(days_sb),
                    "signs": ppicked,
                    "notes": notes,
                }
                with st.spinner("Estimating stage…"):
                    try:
                        out = predict_pregnancy(preg_sym, int(days_sb), pregnancy_bundle, knowledge_base)
                    except Exception as e:
                        out = {"error": str(e)}

                if "error" in out:
                    st.error(out["error"])
                else:
                    st.subheader("Estimated stage")
                    st.metric("Stage", out["stage"])
                    st.progress(min(1.0, max(0.0, out["confidence"] / 100.0)))
                    st.caption(
                        f"Confidence: {out['confidence']}% ({out['confidence_level']}) — {out.get('method', '')}"
                    )
                    st.info("🟡 Pregnancy signs can overlap with illness. Always confirm with your vet (ultrasound / blood test).")
                    st.markdown(out.get("description", ""))
                    st.markdown(f"**Tips:** {out.get('recommendations', '')}")
                    st.markdown(f"**Day-to-day care:** {out.get('care', '')}")
                    if out.get("expected_calving_in"):
                        st.success(f"Rough calving timing from the date you entered: **{out['expected_calving_in']}**")
                    with st.expander("Alternative stages"):
                        for alt in out.get("alternatives", []):
                            st.write(f"- {alt['stage']}: {alt['probability']}%")

                    entry = {
                        "type": "pregnancy",
                        "time_utc": datetime.utcnow().isoformat(),
                        "context": ctx,
                        "result": out,
                    }
                    st.session_state.last_result = entry
                    add_history(entry)

    with tab_hist:
        st.markdown("Last **5** results in this browser session (memory only — nothing is saved online).")
        if not st.session_state.history:
            st.info("No history yet. Run a check in the Symptom checker tab.")
        else:
            for i, h in enumerate(st.session_state.history):
                label = "Illness" if h.get("type") == "disease" else "Pregnancy"
                with st.expander(f"{i + 1}. {label} — {h.get('time_utc', '')}"):
                    st.json(h)

        report = build_report_text(st.session_state.last_result)
        if st.session_state.last_image:
            report += f"\n\nPhoto attached this session: {st.session_state.last_image.get('name', 'photo')}"

        st.download_button(
            "Download report (.txt)",
            data=report,
            file_name="vetai_visit_summary.txt",
            mime="text/plain",
            help="A simple text file you can email or print for your veterinarian.",
        )

    render_disclaimer()


if __name__ == "__main__":
    main()
