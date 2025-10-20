import joblib, os, json, spacy
import pandas as pd
import nlp
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

print("SickScan CLI")
print("Version 67.69.00")
print("Developed by Meowl (^.^)\n")

modelPath = "disease_model.pkl"
encoderPath = "label_encoder.pkl"
symptomsPath = "symptoms_columns.pkl"

if os.path.exists(modelPath) and os.path.exists(encoderPath) and os.path.exists(symptomsPath):
    print("Loading Saved Model...")
    model = joblib.load(modelPath)
    le = joblib.load(encoderPath)
    allSymptoms = joblib.load(symptomsPath)
else:
    print("Ouno!!! Model not found. Please Train it :)")
    exit()

try:
    with open("symptoms_final.json", "r") as f:
        symptomsDict = json.load(f)
except FileNotFoundError:
    print("Ouno!!! Symptoms file not found. Please Get it :)")
    exit()

try:
    with open("disease_symptom_dict.json", "r") as f:
        diseaseSymptomDict = json.load(f)
except FileNotFoundError:
    print("Ouno!!! Disease–Symptom mapping file not found. Please Train again :)")
    exit()

try:
    print("Loading NLP Model...")
    nlpModel = spacy.load("en_core_sci_lg")
except OSError:
    print("scispaCy model not found. Please install it:")
    print("pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz")
    exit()

# --- Main Loop ---
while True:
    text = input("\nHow are you feeling today? ")
    if text.lower().strip() == "exit":
        print("\nGoodbye Twin!")
        print("Stay healthy and SYBAU!!!")
        break

    # Extract symptoms
    extracted = nlp.extract_symptoms_from_text(text, symptomsDict, nlpModel)
    if not extracted:
        print("No symptoms detected. Try describing more about how you feel.")
        continue

    detectedSymptomIDs = [sid for sid, conf in extracted]

    print("\nDetected Symptoms:")
    for sid, conf in extracted:
        name = symptomsDict.get(sid)
        primary_name = name[0] if isinstance(name, list) else name
        print(f"  - {primary_name:<30} (Confidence: {conf}%)")

    # Build first prediction
    userData = pd.DataFrame(0, index=[0], columns=allSymptoms)
    for s in detectedSymptomIDs:
        if s in userData.columns:
            userData.loc[0, s] = 1

    probs = model.predict_proba(userData)[0]
    sortedIndices = probs.argsort()[::-1]
    top5 = [(le.inverse_transform([i])[0], probs[i] * 100) for i in sortedIndices[:5]]

    print("\nTop 5 Possible Diseases:")
    for idx, (disease, confidence) in enumerate(top5, 1):
        syms = ", ".join(sorted(list(diseaseSymptomDict.get(disease, [])))) if disease in diseaseSymptomDict else "N/A"
        print(f"{idx}. {disease:30s} — {confidence:.2f}% confidence")
        print(f"   Common Symptoms: {syms}\n")

    print("You can now pick more symptoms that match how you feel from the above diseases.")
    print("Enter them separated by commas, or press Enter to skip.")
    extraInput = input("→ Additional symptoms: ").strip()

    if extraInput:
        extraSymptoms = [
            s.strip().lower().replace(" ", "_") for s in extraInput.split(",") if s.strip() != ""
        ]
        for s in extraSymptoms:
            if s in userData.columns:
                userData.loc[0, s] = 1
                if s not in detectedSymptomIDs:
                    detectedSymptomIDs.append(s)

    probs_final = model.predict_proba(userData)[0]
    sortedIndices_final = probs_final.argsort()[::-1]
    top3 = [(le.inverse_transform([i])[0], probs_final[i] * 100) for i in sortedIndices_final[:3]]

    print("\nFinal 3 Most Probable Diseases:")
    for disease, confidence in top3:
        syms = ", ".join(sorted(list(diseaseSymptomDict.get(disease, [])))) if disease in diseaseSymptomDict else "N/A"
        print(f"{disease:30s} — {confidence:.2f}% confidence")
        print(f"Common Symptoms: {syms}\n")

    print("──────────────────────────────────────────────")