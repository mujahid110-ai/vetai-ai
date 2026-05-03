/** Human-readable labels (UI only; same text as Streamlit). */

export const DISEASE_SYMPTOM_LABELS = {
  fever: "Fever or high temperature",
  cough: "Coughing",
  nasal_discharge: "Runny nose / nasal discharge",
  diarrhea: "Diarrhea",
  lethargy: "Lethargy (very tired / weak)",
  loss_of_appetite: "Loss of appetite",
  weight_loss: "Weight loss",
  lameness: "Lameness (limping)",
  swelling: "Swelling on the body or legs",
  skin_lesions: "Blisters, sores, or skin lesions",
  breathing_difficulty: "Trouble breathing",
  eye_discharge: "Eye discharge or squinting",
  salivation: "Drooling / extra saliva",
  bloat: "Bloat (very round belly, uncomfortable)",
  pale_mucous_membranes: "Pale gums or inner eyelids",
  milk_decrease: "Drop in milk production",
  reproductive_issues: "Reproductive problems (abortion, discharge, etc.)",
  nervous_signs: "Nervous signs (trembling, circling, unusual behavior)",
  rough_coat: "Rough or dull coat",
  dehydration: "Dehydration (sunken eyes, dry mouth)",
  mucus_discharge: "Mucus discharge",
  restlessness: "Restlessness",
  isolation: "Keeping away from the herd",
  udder_swelling: "Udder swelling or hardness",
  muscle_tremors: "Muscle tremors or shaking",
  foul_discharge: "Foul-smelling discharge",
  recumbency: "Down and cannot stand",
  sweet_breath: "Sweet or fruity breath smell",
  staggering: "Staggering or wobbly walking",
};

export const PREGNANCY_FEATURE_LABELS = {
  cessation_of_estrus: "Heat cycles have stopped (cessation of estrus)",
  calmer_temperament: "Calmer temperament than usual",
  increased_appetite: "Increased appetite",
  weight_gain: "Gradual weight gain",
  vulva_changes: "Vulva looks fuller or softer",
  mucus_discharge: "Clear or stringy mucus from the vulva",
  uterine_asymmetry: "One uterine horn feels larger (needs exam)",
  abdominal_swelling_right: "Right side of belly looks fuller (rumen vs. pregnancy)",
  udder_development: "Udder is developing or filling",
  milk_vein_development: "Milk veins more visible",
  fetal_movement: "Possible fetal movement felt on the flank",
  pelvic_relaxation: "Pelvis feels wider / ligaments looser",
  tail_ligament_softening: "Tail head looks raised; ligaments feel softer",
  isolation_behavior: "Keeps away from the group",
  restlessness: "Restless before calving",
  muscle_tremors: "Muscle tremors (also seen with metabolic disease)",
  colostrum_discharge: "Thick yellow fluid (colostrum) at teats",
  rough_coat: "Rough coat (non-specific)",
};

export function normalizeSymptomKey(col) {
  return col
    .toLowerCase()
    .replace(/\s+/g, "_")
    .replace(/\(/g, "")
    .replace(/\)/g, "");
}

export function diseaseColumnLabel(col) {
  const key = normalizeSymptomKey(col);
  return DISEASE_SYMPTOM_LABELS[key] || col.replace(/_/g, " ");
}

export function pregnancyColumnLabel(col) {
  return PREGNANCY_FEATURE_LABELS[col] || col.replace(/_/g, " ");
}
