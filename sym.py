import pandas as pd
import json

# Load and preprocess dataset
df = pd.read_csv("dataset.csv")
df = df.fillna("none")

# get symptom columns
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

with open("symptoms.json", "w") as f:
    json.dump(symptomsDict, f, indent=4)