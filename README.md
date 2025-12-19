# SickScan 🩺

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3-lightgrey?logo=flask&logoColor=black)](https://flask.palletsprojects.com/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![spaCy](https://img.shields.io/badge/spaCy-3.6-purple?logo=spacy&logoColor=white)](https://spacy.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 1. Introduction

**SickScan** is a web-based healthcare application designed to make preliminary medical diagnosis more accessible and user-friendly. Traditional symptom checkers rely on dropdown menus, which often require users to know medical terminology, leading to inaccurate symptom reporting and frustration.  

SickScan allows users to describe their symptoms in **natural, everyday language**, e.g., “I have a severe headache and feel unusually cold.” Using **biomedical NLP** and **Fuzzy Logic**, the system extracts relevant symptoms and predicts the top five possible diseases with confidence scores, providing early awareness before consulting a medical professional.

---

## 2. Project Flow & System Architecture

### A. Training Phase
1. **Data Ingestion**  
   - Loads `dataset.csv` containing disease-symptom mappings.  
2. **Data Preprocessing**  
   - Handles missing values, label-encodes diseases, and converts symptoms into multi-hot encoded vectors.  
3. **Model Training**  
   - Trains a **Multinomial Naive Bayes** classifier to learn symptom-disease relationships.  
4. **Artifact Generation**  
   - Serializes trained model, label encoders, and symptom mappings into `.pkl` and `.json` files for web deployment.

### B. Inference Phase
1. **User Input**  
   - Users submit a symptom description via the `/analyze` endpoint.  
2. **NLP-Based Symptom Extraction**  
   - Processes text using **scispaCy** and applies **fuzzy string matching** (`TheFuzz`) to detect symptoms even with spelling errors.  
3. **Symptom Mapping**  
   - Converts extracted symptoms into a binary vector for the trained model.  
4. **Disease Prediction**  
   - Calculates disease probabilities using the Naive Bayes classifier.  
5. **Result Presentation**  
   - Returns **top 5 predicted diseases** with confidence scores and commonly associated symptoms.

---

## 3. Technology Stack

### Backend & Frameworks
- **Python** – Core language  
- **Flask** – Web API & interface handling  

### Machine Learning & Data Processing
- **Scikit-learn** – Multinomial Naive Bayes & label encoding  
- **Pandas** – Dataset manipulation & feature engineering  
- **Joblib** – Model serialization  

### Natural Language Processing
- **spaCy (scispaCy)** – Biomedical NLP  
- **TheFuzz** – Fuzzy string matching for symptom extraction  

---

## 4. Models & Dataset

### NLP Model
- **Name:** `en_core_sci_lg`  
- **Description:** Large biomedical English model from **scispaCy**, trained on scientific & medical text.  

### Prediction Model
- **Algorithm:** Multinomial Naive Bayes  
- **Reasoning:** Efficient with categorical/binary features, provides probabilistic outputs, ideal for disease prediction.  

### Dataset
- **Source:** [Kaggle: Disease Symptom Description Dataset](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset?select=dataset.csv)  
- **Structure:** Each row contains a diagnosed disease and associated symptoms.  

---

## 5. Usage

1. Clone the repository:
```bash
git clone https://github.com/SlyHammad18/sickscan.git
cd sickscan
pip install -r requirements.txt
python app.py
```

2. Train the Model
```bash
python train.py
```
