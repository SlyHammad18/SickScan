import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("dataset.csv")    # load the csv file
df = df.fillna("none")             # fills all empty cols with none

# get symptom cols
symptomCols = [c for c in df.columns if c != "Disease"] 
symptoms = sorted(                                        # sorted -> sorts the symptoms in alphabetical order
    set([s.strip().lower().replace(" ", "_")              # set -> removes duplicates
         for s in df[symptomCols].values.flatten()        # flatten -> flattens the array (multidimensional to 1D)
         if s != 'none']
    )
)
# User Friendly Symptoms array, title() capitalizes first letter of each word
symptomsUser = [symptom.replace("_", " ").title() for symptom in symptoms]
# creating a dictionary to map symptoms to their user friendly names
symptomsDict = dict(zip(symptoms, symptomsUser))

# get disease cols
diseaseCols = [c for c in df.columns if c == "Disease"]
diseases = sorted(
    set([d.strip().title() if not d.isupper() else d.strip()    # if all letters are cap dont call title()
         for d in df[diseaseCols].values.flatten() 
         if d != "none"]
    )
)

# create a multi-hot encoded dataframe
multiHot = pd.DataFrame(0, index=range(len(df)), columns=symptoms)
for i, row in df.iterrows():             # looks thorugh each row in dataset
    for s in symptomCols:
        val = row[s].strip().lower().replace(" ", "_")
        if val != 'none' and val in multiHot.columns:
            multiHot.at[i, val] = 1      # if the symptom is in the symptoms array, set the value to 1

# use this line to see the multi-hot encoded dataframe
# multiHot.to_csv("symptoms.csv", index=False)

# encode diseases
le = LabelEncoder()
# Tells the encoder what all possible disease labels are
le.fit(diseases)
# convert each disease name into its numeric label
y = le.transform(df["Disease"].apply(lambda d: d.strip().title() if not d.isupper() else d.strip()))

# train model
model = MultinomialNB()
model.fit(multiHot, y)

print("--- Symptoms List ---")
cols = 4
for i in range(0, len(symptomsUser), cols):
    row = symptomsUser[i:i + cols]
    print("".join(f"{f'{i+j+1}. {s}':40}" for j, s in enumerate(row)))

# take symptoms input
print("Enter at least 5 symptoms (comma-separated):")
user_input = input("→ ").split(",")

# clean user input
user_symptoms = [s.strip().lower().replace(" ", "_") for s in user_input if s.strip() != ""]

# create user data row (multi-hot vector)
userData = pd.DataFrame(0, index=[0], columns=symptoms)
for s in user_symptoms:
    if s in userData.columns:
        userData.loc[0, s] = 1

# make probability predictions for all diseases
probs = model.predict_proba(userData)[0]

# sort probabilities in descending order
sorted_indices = probs.argsort()[::-1]

# get top 3 predictions
top3 = [(le.inverse_transform([i])[0], probs[i] * 100) for i in sorted_indices[:3]]

print("Top 3 Possible Diseases:")
for disease, confidence in top3:
    print(f"{disease:30s} — {confidence:.2f}% confidence")

# highlight top prediction
top_disease, top_conf = top3[0]
print(f"Most likely disease: {top_disease} ({top_conf:.2f}% confidence)")