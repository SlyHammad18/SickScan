import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
import joblib
import os

# ---------- Load and preprocess dataset ----------
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

# get unique diseases
diseaseCols = [c for c in df.columns if c == "Disease"]
diseases = sorted(
    set([
        d.strip().title() if not d.isupper() else d.strip()
        for d in df[diseaseCols].values.flatten()
        if d != "none"
    ])
)

# ---------- Multi-hot encoding ----------
multiHot = pd.DataFrame(0, index=range(len(df)), columns=symptoms)
for i, row in df.iterrows():
    for s in symptomCols:
        val = row[s].strip().lower().replace(" ", "_")
        if val != 'none' and val in multiHot.columns:
            multiHot.at[i, val] = 1

# ---------- Encode diseases ----------
le = LabelEncoder()
le.fit(diseases)
y = le.transform(df["Disease"].apply(lambda d: d.strip().title() if not d.isupper() else d.strip()))

# ---------- Train or Load Model ----------
model_path = "disease_model.joblib"
encoder_path = "label_encoder.joblib"
columns_path = "symptoms_columns.joblib"

if os.path.exists(model_path) and os.path.exists(encoder_path):
    print("✅ Loading saved model...")
    model = joblib.load(model_path)
    le = joblib.load(encoder_path)
    symptoms = joblib.load(columns_path)
else:
    print("⚙️ Training new model...")
    model = MultinomialNB()
    model.fit(multiHot, y)
    joblib.dump(model, model_path)
    joblib.dump(le, encoder_path)
    joblib.dump(symptoms, columns_path)
    print("💾 Model saved successfully!")

# ---------- Build disease-symptom mapping ----------
diseaseSymptomDict = {}
for i, row in df.iterrows():
    diseaseName = row["Disease"].strip().title() if not row["Disease"].isupper() else row["Disease"].strip()
    syms = []
    for s in symptomCols:
        val = row[s].strip().lower().replace(" ", "_")
        if val != 'none':
            syms.append(symptomsDict.get(val, val))
    diseaseSymptomDict.setdefault(diseaseName, set()).update(syms)

# ---------- Prediction Loop ----------
while True:
    print("\n--- Symptoms List ---")
    cols = 4
    for i in range(0, len(symptomsUser), cols):
        row = symptomsUser[i:i + cols]
        print("".join(f"{f'{i+j+1}. {s}':40}" for j, s in enumerate(row)))

    print("\nEnter at least 5 symptoms (comma-separated):")
    userInput = input("→ ").split(",")

    userSymptoms = [s.strip().lower().replace(" ", "_") for s in userInput if s.strip() != ""]
    userData = pd.DataFrame(0, index=[0], columns=symptoms)
    for s in userSymptoms:
        if s in userData.columns:
            userData.loc[0, s] = 1

    probs = model.predict_proba(userData)[0]
    sortedIndices = probs.argsort()[::-1]
    top5 = [(le.inverse_transform([i])[0], probs[i] * 100) for i in sortedIndices[:5]]

    print("\nTop 3 Possible Diseases:")
    for disease, confidence in top5:
        syms = ", ".join(sorted(list(diseaseSymptomDict.get(disease, []))))
        print(f"{disease:30s} — {confidence:.2f}% confidence")
        print(f"Other Symptoms: {syms}\n")