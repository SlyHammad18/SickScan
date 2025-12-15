import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
import joblib
import json

def train_model():
    print("--- Starting Training Process ---")

    # Load and preprocess dataset
    print("Loading dataset...")
    try:
        df = pd.read_csv("dataset.csv")
    except FileNotFoundError:
        print("Error: dataset.csv not found!")
        return

    df = df.fillna("none")

    # get symptom columns
    print("Extracting symptoms...")
    symptomCols = [c for c in df.columns if c != "Disease"]
    symptoms = sorted(
        set([
            s.strip().lower().replace(" ", "_")
            for s in df[symptomCols].values.flatten()
            if s != 'none'
        ])
    )

    # user-friendly names
    symptomsUser = [symptom.replace("_", " ").title() for symptom in symptoms]
    symptomsDict = dict(zip(symptoms, symptomsUser))

    # get unique diseases
    print("Extracting diseases...")
    diseaseCols = [c for c in df.columns if c == "Disease"]
    diseases = sorted(
        set([
            d.strip().title() if not d.isupper() else d.strip()
            for d in df[diseaseCols].values.flatten()
            if d != "none"
        ])
    )

    # Multi-hot encoding
    print("Encoding data...")
    multiHot = pd.DataFrame(0, index=range(len(df)), columns=symptoms)
    for i, row in df.iterrows():
        for s in symptomCols:
            val = row[s].strip().lower().replace(" ", "_")
            if val != 'none' and val in multiHot.columns:
                multiHot.at[i, val] = 1

    # Encode diseases
    le = LabelEncoder()
    le.fit(diseases)
    y = le.transform(df["Disease"].apply(lambda d: d.strip().title() if not d.isupper() else d.strip()))

    # Train Model
    print("Training Naive Bayes model...")
    model = MultinomialNB()
    model.fit(multiHot, y)

    # Save Artifacts
    model_path = "disease_model.pkl"
    encoder_path = "label_encoder.pkl"
    columns_path = "symptoms_columns.pkl"

    print(f"Saving model to {model_path}...")
    joblib.dump(model, model_path)
    
    print(f"Saving encoder to {encoder_path}...")
    joblib.dump(le, encoder_path)
    
    print(f"Saving symptom columns to {columns_path}...")
    joblib.dump(symptoms, columns_path)

    # Build disease-symptom mapping
    print("Building disease-symptom mapping...")
    diseaseSymptomDict = {}
    for i, row in df.iterrows():
        diseaseName = row["Disease"].strip().title() if not row["Disease"].isupper() else row["Disease"].strip()
        syms = []
        for s in symptomCols:
            val = row[s].strip().lower().replace(" ", "_")
            if val != 'none':
                syms.append(symptomsDict.get(val, val))
        diseaseSymptomDict.setdefault(diseaseName, set()).update(syms)

    # Save disease-symptom mapping
    mapping_path = "disease_symptom_dict.json"
    print(f"Saving disease-symptom mapping to {mapping_path}...")
    with open(mapping_path, "w") as f:
        json.dump({k: list(v) for k, v in diseaseSymptomDict.items()}, f, indent=4)

    print("--- Training Completed Successfully ---")

if __name__ == "__main__":
    train_model()