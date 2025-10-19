import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB

# Filenames for caching
MODEL_FILE = "disease_model.pkl"
ENCODER_FILE = "label_encoder.pkl"
SYMPTOMS_FILE = "symptoms_list.pkl"

print("Loading Dataset...")
df = pd.read_csv("dataset.csv")  
df = df.fillna(0)  # replace missing values with 0 (since 0/1 data)
print("Dataset Loaded")

# Separate features and target
X = df.drop(columns=["diseases"])  # all symptom columns (0/1)
y = df["diseases"]                 # disease labels
symptoms = list(X.columns)

# If model exists, load it to save time
if os.path.exists(MODEL_FILE) and os.path.exists(ENCODER_FILE) and os.path.exists(SYMPTOMS_FILE):
    print("✅ Loading existing trained model...")
    model = joblib.load(MODEL_FILE)
    le = joblib.load(ENCODER_FILE)
    symptoms = joblib.load(SYMPTOMS_FILE)
else:
    print("Encoding...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print("Done")

    print("Training...")
    model = MultinomialNB()
    model.fit(X, y_encoded)
    print("✅ Training Done")

    # Save for future use
    joblib.dump(model, MODEL_FILE)
    joblib.dump(le, ENCODER_FILE)
    joblib.dump(symptoms, SYMPTOMS_FILE)
    print("💾 Model, encoder, and symptoms saved successfully!")

# Dictionary: disease → symptoms
diseaseSymptomDict = {}
for disease, group in df.groupby("diseases"):
    symptomList = [s for s in symptoms if group[s].sum() > 0]
    diseaseSymptomDict[disease] = symptomList

# Interactive prediction loop
while True:
    print("\n--- Symptom List ---")
    cols = 4
    for i in range(0, len(symptoms), cols):
        row = symptoms[i:i + cols]
        print("".join(f"{f'{i+j+1}. {s.replace('_', ' ').title()}':40}" for j, s in enumerate(row)))

    print("\nEnter your symptoms (comma-separated):")
    userInput = input("→ ").split(",")

    # Clean user input
    userSymptoms = [s.strip().lower().replace(" ", "_") for s in userInput if s.strip() != ""]

    # Create a user data row (multi-hot vector)
    userData = pd.DataFrame(0, index=[0], columns=symptoms)
    for s in userSymptoms:
        if s in userData.columns:
            userData.loc[0, s] = 1

    # Predict probabilities
    probs = model.predict_proba(userData)[0]
    sortedIndices = probs.argsort()[::-1]

    # Top 5 possible diseases
    top10 = [(le.inverse_transform([i])[0], probs[i] * 100) for i in sortedIndices[:10]]

    print("\nTop 10 Possible Diseases:")
    for disease, confidence in top10[:10]:
        syms = ", ".join(sorted(list(diseaseSymptomDict.get(disease, []))))
        print(f"{disease:40s} — {confidence:.2f}% confidence")
        print(f"Common Symptoms: {syms}\n")