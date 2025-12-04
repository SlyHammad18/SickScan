import joblib
import os
import json
import spacy
import pandas as pd
import nlp
import warnings

# Suppress warnings as per original code
warnings.filterwarnings("ignore", category=FutureWarning)

# Constants
MODEL_PATH = "disease_model.pkl"
ENCODER_PATH = "label_encoder.pkl"
SYMPTOMS_PATH = "symptoms_columns.pkl"
SYMPTOMS_DICT_PATH = "symptoms_final.json"
DISEASE_SYMPTOM_DICT_PATH = "disease_symptom_dict.json"
NLP_MODEL_NAME = "en_core_sci_lg"

def load_artifacts():
    """Loads all necessary models and data files."""
    if not (os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH) and os.path.exists(SYMPTOMS_PATH)):
        print("Ouno!!! Model not found. Please Train it :)")
        return None, None, None, None, None, None

    print("Loading Saved Model...")
    model = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    all_symptoms = joblib.load(SYMPTOMS_PATH)

    try:
        with open(SYMPTOMS_DICT_PATH, "r") as f:
            symptoms_dict = json.load(f)
    except FileNotFoundError:
        print("Ouno!!! Symptoms file not found. Please Get it :)")
        return None, None, None, None, None, None

    try:
        with open(DISEASE_SYMPTOM_DICT_PATH, "r") as f:
            disease_symptom_dict = json.load(f)
    except FileNotFoundError:
        print("Ouno!!! Disease–Symptom mapping file not found. Please Train again :)")
        return None, None, None, None, None, None

    try:
        print("Loading NLP Model...")
        nlp_model = spacy.load(NLP_MODEL_NAME)
    except OSError:
        print("scispaCy model not found. Please install it:")
        print(f"pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/{NLP_MODEL_NAME}-0.5.4.tar.gz")
        return None, None, None, None, None, None

    return model, le, all_symptoms, symptoms_dict, disease_symptom_dict, nlp_model

def predict_disease(model, le, all_symptoms, detected_symptoms_ids):
    """Predicts disease based on detected symptoms."""
    user_data = pd.DataFrame(0, index=[0], columns=all_symptoms)
    for s in detected_symptoms_ids:
        if s in user_data.columns:
            user_data.loc[0, s] = 1

    probs = model.predict_proba(user_data)[0]
    sorted_indices = probs.argsort()[::-1]
    
    # Return top predictions as list of (disease_name, probability_percentage)
    return [(le.inverse_transform([i])[0], probs[i] * 100) for i in sorted_indices]

def print_predictions(predictions, disease_symptom_dict, top_n=5):
    """Prints the top N disease predictions."""
    for idx, (disease, confidence) in enumerate(predictions[:top_n], 1):
        syms = ", ".join(sorted(list(disease_symptom_dict.get(disease, [])))) if disease in disease_symptom_dict else "N/A"
        print(f"{idx}. {disease:30s} — {confidence:.2f}% confidence")
        print(f"   Common Symptoms: {syms}\n")

def run_cli():
    """Runs the main interactive CLI loop."""
    print("SickScan CLI")
    print("Version 67.69.00")
    print("Developed by Meowl (^.^)\n")

    model, le, all_symptoms, symptoms_dict, disease_symptom_dict, nlp_model = load_artifacts()
    
    if model is None:
        return # Exit if loading failed

    while True:
        text = input("\nHow are you feeling today? ")
        if text.lower().strip() == "exit":
            print("\nGoodbye Twin!")
            print("Stay healthy and SYBAU!!!")
            break

        # Extract symptoms
        extracted = nlp.extract_symptoms_from_text(text, symptoms_dict, nlp_model)
        if not extracted:
            print("No symptoms detected. Try describing more about how you feel.")
            continue

        detected_symptom_ids = [sid for sid, conf in extracted]

        print("\nDetected Symptoms:")
        for sid, conf in extracted:
            name = symptoms_dict.get(sid)
            primary_name = name[0] if isinstance(name, list) else name
            print(f"  - {primary_name:<30} (Confidence: {conf}%)")

        # First prediction
        top_predictions = predict_disease(model, le, all_symptoms, detected_symptom_ids)
        
        print("\nTop 5 Possible Diseases:")
        print_predictions(top_predictions, disease_symptom_dict, top_n=5)

        print("You can now pick more symptoms that match how you feel from the above diseases.")
        print("Enter them separated by commas, or press Enter to skip.")
        extra_input = input("→ Additional symptoms: ").strip()

        if extra_input:
            extra_symptoms = [
                s.strip().lower().replace(" ", "_") for s in extra_input.split(",") if s.strip() != ""
            ]
            # Add valid extra symptoms to our list
            for s in extra_symptoms:
                 if s in all_symptoms and s not in detected_symptom_ids:
                     detected_symptom_ids.append(s)

            # Re-predict with extra symptoms
            final_predictions = predict_disease(model, le, all_symptoms, detected_symptom_ids)
            
            print("\nFinal 3 Most Probable Diseases:")
            # Note: The original code printed top 3 here without the index number, just disease and confidence
            for disease, confidence in final_predictions[:3]:
                syms = ", ".join(sorted(list(disease_symptom_dict.get(disease, [])))) if disease in disease_symptom_dict else "N/A"
                print(f"{disease:30s} — {confidence:.2f}% confidence")
                print(f"Common Symptoms: {syms}\n")

        print("──────────────────────────────────────────────")

if __name__ == "__main__":
    run_cli()