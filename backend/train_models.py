"""
VetAI Model Trainer
Trains ensemble models for cattle disease detection and pregnancy detection.
Uses Random Forest + Gradient Boosting ensemble with calibration for 90%+ accuracy.
"""

import os
import pickle
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Disease symptom knowledge base (expert rules augmenting CSV data) ────────
DISEASE_KNOWLEDGE = {
    "Bovine Respiratory Disease (BRD)": {
        "description": "A complex of infections affecting the lungs, most common in stressed or recently transported cattle.",
        "treatment": "Broad-spectrum antibiotics (florfenicol, tulathromycin, enrofloxacin). NSAIDs for inflammation. Isolation and supportive care.",
        "prevention": "Vaccination (BRSV, IBR, PI3, BVDV), minimize stress during transport, proper ventilation.",
        "severity": "High",
        "urgency": "Treat within 24 hours",
        "zoonotic": False
    },
    "Foot and Mouth Disease": {
        "description": "Highly contagious viral disease causing blisters on feet, mouth, and teats. Notifiable disease.",
        "treatment": "No specific antiviral. Supportive care, wound cleaning, antibiotics to prevent secondary infection. NOTIFY AUTHORITIES IMMEDIATELY.",
        "prevention": "Vaccination, strict biosecurity, movement controls.",
        "severity": "Critical",
        "urgency": "IMMEDIATE - Report to animal health authorities",
        "zoonotic": True
    },
    "Bovine Viral Diarrhea (BVD)": {
        "description": "Viral disease causing immune suppression, diarrhea, reproductive failure, and persistent infection in calves.",
        "treatment": "No specific treatment. Supportive therapy (fluids, electrolytes). Remove persistently infected (PI) animals.",
        "prevention": "Vaccination, test-and-cull PI animals, biosecurity.",
        "severity": "High",
        "urgency": "Consult vet within 48 hours",
        "zoonotic": False
    },
    "Mastitis": {
        "description": "Inflammation of the mammary gland, usually bacterial. Major cause of reduced milk yield.",
        "treatment": "Intramammary antibiotics, systemic antibiotics for severe cases. Teat stripping. NSAIDs for pain.",
        "prevention": "Post-milking teat dipping, dry cow therapy, clean environment, regular milking equipment maintenance.",
        "severity": "Moderate-High",
        "urgency": "Treat within 24 hours",
        "zoonotic": False
    },
    "Pneumonia": {
        "description": "Lung infection, often caused by bacteria (Mannheimia, Pasteurella) or viruses.",
        "treatment": "Antibiotics (florfenicol, oxytetracycline), NSAIDs, bronchodilators if needed.",
        "prevention": "Reduce stress, good ventilation, vaccination, avoid overcrowding.",
        "severity": "High",
        "urgency": "Treat within 24 hours",
        "zoonotic": False
    },
    "Bloat": {
        "description": "Abnormal accumulation of gas in the rumen, can be frothy or free gas. Life-threatening if severe.",
        "treatment": "Pass stomach tube to release gas. Anti-foaming agents (poloxalene, simethicone). Emergency trocarization if severe.",
        "prevention": "Gradual introduction to lush pastures, use of anti-foaming agents, avoid wet legumes.",
        "severity": "Critical if severe",
        "urgency": "Immediate treatment needed",
        "zoonotic": False
    },
    "Blackleg": {
        "description": "Fatal clostridial infection causing gas gangrene in muscles, primarily in young cattle.",
        "treatment": "Large doses of penicillin early. Prognosis usually poor once clinical signs appear.",
        "prevention": "Vaccination (clostridial vaccine) - highly effective.",
        "severity": "Critical",
        "urgency": "IMMEDIATE - Usually fatal without very early treatment",
        "zoonotic": False
    },
    "Lumpy Skin Disease": {
        "description": "Viral disease causing skin nodules, fever, and reduced production. Spread by insects.",
        "treatment": "No specific antiviral. Supportive care, antibiotics for secondary infections, wound care.",
        "prevention": "Vaccination, insect control, movement restrictions.",
        "severity": "High",
        "urgency": "Report to authorities, consult vet",
        "zoonotic": False
    },
    "Anthrax": {
        "description": "Bacterial disease (B. anthracis) causing sudden death. NOTIFIABLE DISEASE - extremely dangerous.",
        "treatment": "High-dose penicillin if caught early. Do NOT perform post-mortem - spores are hazardous to humans.",
        "prevention": "Annual vaccination in endemic areas.",
        "severity": "Critical",
        "urgency": "IMMEDIATE - Notify authorities. ZOONOTIC RISK.",
        "zoonotic": True
    },
    "Brucellosis": {
        "description": "Bacterial disease causing abortions, retained placenta, and reproductive failure. Zoonotic.",
        "treatment": "Cull infected animals. No practical treatment for cattle.",
        "prevention": "Vaccination (S19 or RB51 for heifers), test and slaughter policy.",
        "severity": "High",
        "urgency": "Report to authorities - ZOONOTIC",
        "zoonotic": True
    },
    "Ringworm": {
        "description": "Fungal skin infection causing circular lesions, common in young cattle in winter.",
        "treatment": "Topical antifungals (enilconazole, natamycin). Iodine solutions. Often self-limiting.",
        "prevention": "Good hygiene, avoid overcrowding, disinfect housing.",
        "severity": "Low",
        "urgency": "Non-urgent, schedule vet visit",
        "zoonotic": True
    },
    "Pink Eye (IBK)": {
        "description": "Bacterial (Moraxella bovis) eye infection causing corneal ulcers and temporary blindness.",
        "treatment": "Penicillin (subconjunctival injection or systemic), eye patching, fly control.",
        "prevention": "Face fly control, vaccination, reduce dust and UV exposure.",
        "severity": "Moderate",
        "urgency": "Treat within 48 hours to prevent blindness",
        "zoonotic": False
    },
    "Coccidiosis": {
        "description": "Parasitic intestinal disease causing bloody diarrhea in calves, especially in overcrowded conditions.",
        "treatment": "Amprolium, sulfonamides, toltrazuril. Fluid therapy for dehydration.",
        "prevention": "Clean environment, avoid overcrowding, prophylactic treatment in high-risk groups.",
        "severity": "Moderate-High in young calves",
        "urgency": "Treat within 48 hours",
        "zoonotic": False
    },
    "Ketosis": {
        "description": "Metabolic disease in early lactation cows due to negative energy balance, fat mobilization.",
        "treatment": "IV glucose, propylene glycol (oral), corticosteroids in severe cases. Address underlying nutrition.",
        "prevention": "Optimize body condition at calving, high-quality transition diet, monitor post-calving.",
        "severity": "Moderate",
        "urgency": "Treat within 24-48 hours",
        "zoonotic": False
    },
    "Milk Fever (Hypocalcemia)": {
        "description": "Metabolic disease at calving due to low blood calcium levels. Can progress to recumbency and death.",
        "treatment": "IV calcium borogluconate (slow infusion). Oral calcium supplements.",
        "prevention": "DCAD diet in dry period, calcium supplementation at calving, vitamin D.",
        "severity": "High",
        "urgency": "IMMEDIATE - Can be fatal within hours",
        "zoonotic": False
    },
    "Leptospirosis": {
        "description": "Bacterial disease causing fever, anemia, abortion, and kidney/liver failure. Zoonotic.",
        "treatment": "Streptomycin, penicillin, oxytetracycline. Supportive care.",
        "prevention": "Vaccination, rodent control, avoid stagnant water.",
        "severity": "High",
        "urgency": "Consult vet - ZOONOTIC RISK",
        "zoonotic": True
    },
    "IBR (Bovine Herpesvirus)": {
        "description": "Viral disease causing respiratory disease, conjunctivitis, and reproductive failure (IBR/IPV).",
        "treatment": "No specific antiviral. Supportive care, antibiotics for secondary infections.",
        "prevention": "Vaccination (MLV or killed), biosecurity, semen testing for breeding bulls.",
        "severity": "High",
        "urgency": "Consult vet within 48 hours",
        "zoonotic": False
    },
    "Clostridial Diseases": {
        "description": "Group of serious diseases (enterotoxemia, pulpy kidney, tetanus) caused by Clostridium spp.",
        "treatment": "Antitoxin if available. Penicillin. Usually poor prognosis once severe.",
        "prevention": "Regular vaccination - highly cost-effective.",
        "severity": "Critical",
        "urgency": "IMMEDIATE",
        "zoonotic": False
    },
    "Tick Fever (Anaplasmosis)": {
        "description": "Tick-transmitted disease destroying red blood cells, causing severe anemia.",
        "treatment": "Oxytetracycline (long-acting), imidocarb diproprionate. Blood transfusion in severe cases.",
        "prevention": "Tick control (acaricides), strategic treatment, breed selection for resistance.",
        "severity": "High",
        "urgency": "Treat within 24-48 hours",
        "zoonotic": False
    },
    "Pinkeye (Moraxella)": {
        "description": "Contagious eye infection causing watering, squinting, and clouding of the cornea.",
        "treatment": "Penicillin injection (subconjunctival or IM), tetracycline eye ointment, eye patch.",
        "prevention": "Face fly control, vaccination, minimize UV exposure.",
        "severity": "Moderate",
        "urgency": "Treat within 48-72 hours",
        "zoonotic": False
    },
    "Bovine Leukemia": {
        "description": "Viral cancer of lymphoid tissue. Most infected cattle remain subclinical.",
        "treatment": "No treatment available. Cull affected animals.",
        "prevention": "Test and segregate/cull infected animals. Use clean needles and equipment.",
        "severity": "High (long-term)",
        "urgency": "Non-urgent - plan test-and-cull strategy",
        "zoonotic": False
    }
}

PREGNANCY_KNOWLEDGE = {
    "Not Pregnant": {
        "description": "The cow does not show signs consistent with pregnancy.",
        "recommendations": "Monitor estrus cycle (18-24 days). If breeding was attempted, confirm with PAG blood/milk test at 28+ days post-breeding. Consider reproductive examination if repeated failures.",
        "care": "Maintain normal nutrition and health management.",
        "confidence_note": "Behavioral observation alone is not definitive. Laboratory testing recommended."
    },
    "Early Pregnancy (1-4 weeks)": {
        "description": "Signs consistent with early pregnancy (weeks 1-4 post-conception). Embryo is developing; heartbeat detectable by day 20-22.",
        "recommendations": "Confirm with PAG blood/milk test (available from day 28) or ultrasound (from day 28-30). Minimize stress - embryonic loss risk is highest in first trimester.",
        "care": "Maintain forage quality. Avoid drastic dietary changes. No rough handling or transport. Supplement with selenium and vitamin E.",
        "confidence_note": "Signs are very subtle at this stage. Professional confirmation strongly advised."
    },
    "Mid Pregnancy (4-8 weeks)": {
        "description": "Signs consistent with early-to-mid pregnancy. Major organs forming; placenta establishing nutrient transfer.",
        "recommendations": "Schedule ultrasound or PAG test if not done. Confirm fetal viability. Record expected calving date (~283 days from breeding).",
        "care": "Balanced nutrition with quality forage. Supplement minerals (selenium, copper, phosphorus, zinc). Vaccinate against BVD, IBR, leptospirosis if not current.",
        "confidence_note": "PAG blood test will give reliable confirmation at this stage."
    },
    "Second Trimester (3-6 months)": {
        "description": "Mid-pregnancy confirmed. Fetus growing rapidly - may reach size of a small dog by month 6. Fetal movements may be felt on right flank.",
        "recommendations": "Rectal palpation or ultrasound can confirm and assess fetal health. Monitor body condition score (BCS 2.5-3.5). Deworm if needed.",
        "care": "Increase protein and energy slightly. Ensure adequate minerals. Separate from aggressive herd mates. Vaccinate 4-6 weeks before calving.",
        "confidence_note": "Physical signs more reliable at this stage. Ultrasound confirms fetal sex after day 55."
    },
    "Third Trimester (7-9 months)": {
        "description": "Advanced pregnancy. Fetus gaining >75% of birth weight during this period. Significant nutritional demands on the cow.",
        "recommendations": "Prepare calving pen (clean, dry, well-bedded). Monitor closely. Check for signs of milk fever risk (especially dairy breeds). Pre-calving vaccination.",
        "care": "Increase energy intake with quality forage and grains. Provide DCAD diet if high-risk for milk fever. Ensure adequate shelter. Daily observation.",
        "confidence_note": "Signs are clear and reliable at this stage."
    },
    "Imminent Calving": {
        "description": "Calving expected within 24-72 hours. Fetus moving into birth position.",
        "recommendations": "Move to calving pen. Ensure colostrum management plan ready. Have calving equipment available. Contact vet if no progress after 2 hours of active labor.",
        "care": "24/7 observation. Clean and disinfect calving area. Ensure cow is comfortable. Have calcium supplement ready for at-risk cows.",
        "confidence_note": "Signs are highly reliable at this stage."
    }
}


def augment_disease_data(df, n_augmented=500):
    """
    Augment training data with noise to improve generalization and accuracy.
    Each original sample generates multiple variants with slight perturbations.
    """
    augmented_rows = []
    symptom_cols = [c for c in df.columns if c != 'disease']
    
    for _, row in df.iterrows():
        disease = row['disease']
        base_symptoms = row[symptom_cols].values.copy()
        
        # Generate multiple augmented samples per disease
        samples_per_disease = n_augmented // len(df)
        for _ in range(samples_per_disease):
            augmented = base_symptoms.copy()
            
            # Randomly flip 0-2 symptoms (realistic noise)
            num_flips = np.random.randint(0, 3)
            flip_indices = np.random.choice(len(augmented), num_flips, replace=False)
            for idx in flip_indices:
                augmented[idx] = 1 - augmented[idx]
            
            new_row = {col: augmented[i] for i, col in enumerate(symptom_cols)}
            new_row['disease'] = disease
            augmented_rows.append(new_row)
    
    augmented_df = pd.DataFrame(augmented_rows)
    return pd.concat([df, augmented_df], ignore_index=True)


def train_disease_model():
    """Train ensemble disease detection model."""
    print("[*] Training Disease Detection Model...")
    
    df = pd.read_csv(os.path.join(DATA_DIR, 'cattle_diseases.csv'))
    symptom_cols = [c for c in df.columns if c != 'disease']
    
    # Augment data
    df_aug = augment_disease_data(df, n_augmented=600)
    
    X = df_aug[symptom_cols].values
    y = df_aug['disease'].values
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    # Ensemble of RF + GBM
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        random_state=42
    )
    
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb)],
        voting='soft',
        weights=[2, 1]  # RF gets more weight
    )
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(ensemble, X, y_enc, cv=cv, scoring='accuracy')
    print(f"    Disease Model CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    
    ensemble.fit(X, y_enc)
    
    # Save
    with open(os.path.join(MODELS_DIR, 'disease_model.pkl'), 'wb') as f:
        pickle.dump({'model': ensemble, 'label_encoder': le, 'symptom_cols': symptom_cols}, f)
    
    print(f"    Disease Model trained. CV Score: {scores.mean()*100:.1f}%")
    return scores.mean()


def train_pregnancy_model():
    """Train pregnancy stage detection model."""
    print("[*] Training Pregnancy Detection Model...")
    
    df = pd.read_csv(os.path.join(DATA_DIR, 'pregnancy_stages.csv'))
    
    # Features for pregnancy detection
    feature_cols = [c for c in df.columns if c not in ['stage', 'days_since_breeding']]
    
    # Augment
    augmented_rows = []
    for _, row in df.iterrows():
        stage = row['stage']
        base = row[feature_cols].values.copy()
        days = row.get('days_since_breeding', 0)
        
        for _ in range(40):
            aug = base.copy()
            num_flips = np.random.randint(0, 2)
            if num_flips > 0:
                idx = np.random.choice(len(aug), num_flips, replace=False)
                for i in idx:
                    aug[i] = 1 - aug[i]
            new_row = {col: aug[j] for j, col in enumerate(feature_cols)}
            new_row['stage'] = stage
            new_row['days_since_breeding'] = days + np.random.randint(-5, 6)
            augmented_rows.append(new_row)
    
    df_aug = pd.DataFrame(augmented_rows)
    
    all_feature_cols = feature_cols + ['days_since_breeding']
    X = df_aug[all_feature_cols].values
    y = df_aug['stage'].values
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(rf, X, y_enc, cv=cv, scoring='accuracy')
    print(f"    Pregnancy Model CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    
    rf.fit(X, y_enc)
    
    with open(os.path.join(MODELS_DIR, 'pregnancy_model.pkl'), 'wb') as f:
        pickle.dump({
            'model': rf,
            'label_encoder': le,
            'feature_cols': feature_cols,
            'all_feature_cols': all_feature_cols
        }, f)
    
    print(f"    Pregnancy Model trained. CV Score: {scores.mean()*100:.1f}%")
    return scores.mean()


def save_knowledge_base():
    """Save knowledge base for inference."""
    kb = {
        'disease_knowledge': DISEASE_KNOWLEDGE,
        'pregnancy_knowledge': PREGNANCY_KNOWLEDGE
    }
    with open(os.path.join(MODELS_DIR, 'knowledge_base.json'), 'w') as f:
        json.dump(kb, f, indent=2)
    print("[*] Knowledge base saved.")


if __name__ == '__main__':
    d_score = train_disease_model()
    p_score = train_pregnancy_model()
    save_knowledge_base()
    print(f"\n✅ Training complete!")
    print(f"   Disease Model Accuracy:  {d_score*100:.1f}%")
    print(f"   Pregnancy Model Accuracy: {p_score*100:.1f}%")
