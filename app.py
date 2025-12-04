import os
import json
import joblib
import pandas as pd
import spacy
from flask import Flask, render_template, request, jsonify
import nlp
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)

# Constants
MODEL_PATH = "disease_model.pkl"
ENCODER_PATH = "label_encoder.pkl"
SYMPTOMS_PATH = "symptoms_columns.pkl"
SYMPTOMS_DICT_PATH = "symptoms_final.json"
DISEASE_SYMPTOM_DICT_PATH = "disease_symptom_dict.json"
NLP_MODEL_NAME = "en_core_sci_lg"

# Global variables to hold loaded artifacts
model = None
le = None
all_symptoms = None
symptoms_dict = None
disease_symptom_dict = None
nlp_model = None

def load_artifacts():
    """Loads all necessary models and data files."""
    global model, le, all_symptoms, symptoms_dict, disease_symptom_dict, nlp_model
    
    print("Loading artifacts...")
    
    if not (os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH) and os.path.exists(SYMPTOMS_PATH)):
        print("Error: Model files not found.")
        return False

    model = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    all_symptoms = joblib.load(SYMPTOMS_PATH)

    try:
        with open(SYMPTOMS_DICT_PATH, "r") as f:
            symptoms_dict = json.load(f)
    except FileNotFoundError:
        print("Error: Symptoms dictionary not found.")
        return False

    try:
        with open(DISEASE_SYMPTOM_DICT_PATH, "r") as f:
            disease_symptom_dict = json.load(f)
    except FileNotFoundError:
        print("Error: Disease-Symptom dictionary not found.")
        return False

    try:
        print("Loading NLP Model...")
        nlp_model = spacy.load(NLP_MODEL_NAME)
    except OSError:
        print(f"Error: scispaCy model '{NLP_MODEL_NAME}' not found.")
        return False

    print("Artifacts loaded successfully!")
    return True

# Load artifacts on startup
load_artifacts()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    extracted = nlp.extract_symptoms_from_text(text, symptoms_dict, nlp_model)
    
    # Format extracted symptoms for frontend
    detected_symptoms = []
    detected_ids = []
    for sid, conf in extracted:
        name = symptoms_dict.get(sid)
        primary_name = name[0] if isinstance(name, list) else name
        detected_symptoms.append({
            'id': sid,
            'name': primary_name,
            'confidence': conf
        })
        detected_ids.append(sid)
        
    return jsonify({
        'symptoms': detected_symptoms,
        'detected_ids': detected_ids
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symptom_ids = data.get('symptoms', [])
    
    if not symptom_ids:
        return jsonify({'error': 'No symptoms provided'}), 400
        
    # Prepare input for model
    user_data = pd.DataFrame(0, index=[0], columns=all_symptoms)
    for s in symptom_ids:
        if s in user_data.columns:
            user_data.loc[0, s] = 1
            
    # Predict
    probs = model.predict_proba(user_data)[0]
    sorted_indices = probs.argsort()[::-1]
    
    top_predictions = []
    for i in sorted_indices[:5]: # Top 5
        disease_name = le.inverse_transform([i])[0]
        confidence = probs[i] * 100
        
        # Get common symptoms for this disease
        common_syms = []
        if disease_name in disease_symptom_dict:
            common_syms = sorted(list(disease_symptom_dict.get(disease_name, [])))
            
        top_predictions.append({
            'disease': disease_name,
            'confidence': round(confidence, 2),
            'common_symptoms': common_syms
        })
        
    return jsonify({'predictions': top_predictions})

@app.route('/all_symptoms', methods=['GET'])
def get_all_symptoms():
    """Returns a list of all available symptoms for autocomplete."""
    # Create a list of {id, name} for all symptoms
    symptom_list = []
    for sid in all_symptoms:
        # Try to find a readable name in the dictionary, otherwise use the ID
        name = sid
        if sid in symptoms_dict:
             val = symptoms_dict[sid]
             name = val[0] if isinstance(val, list) else val
        
        symptom_list.append({'id': sid, 'name': name})
        
    return jsonify(symptom_list)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)