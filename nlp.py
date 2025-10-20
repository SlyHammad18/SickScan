import spacy
import scispacy # Required for the model to load correctly
import json
from thefuzz import fuzz

# --- THE DEFINITIVE FIX: FUZZY MATCHING WITH CONFIDENCE ---

# We only accept a match if the similarity score is above this threshold.
# 85 is a good starting point for high precision.
CONFIDENCE_THRESHOLD = 85

def extract_symptoms_from_text(text, symptom_dict, nlp_model):
    """
    Extracts symptoms using robust fuzzy string matching, returning only high-confidence results.
    """
    # Use a dictionary to store the best match for each symptom to avoid duplicates.
    # Format: {symptom_id: best_confidence_score}
    best_matches = {}

    # 1. Split the user's text into individual sentences.
    doc = nlp_model(text)
    user_sentences = [sent.text for sent in doc.sents]

    # 2. Iterate through every symptom and its known synonyms.
    for symptom_id, value in symptom_dict.items():
        synonym_phrases = value if isinstance(value, list) else [value]
        
        for synonym in synonym_phrases:
            # 3. Compare each synonym against each user sentence.
            for sentence in user_sentences:
                # 4. Calculate the confidence score using a robust fuzzy matching algorithm.
                # token_set_ratio is excellent as it ignores word order and extra words.
                confidence = fuzz.token_set_ratio(synonym.lower(), sentence.lower())

                # 5. If the confidence is high enough, consider it a potential match.
                if confidence >= CONFIDENCE_THRESHOLD:
                    # Check if this symptom has been found before with a lower confidence.
                    # If this match is better, update it.
                    if symptom_id not in best_matches or confidence > best_matches[symptom_id]:
                        best_matches[symptom_id] = confidence

    # Convert the dictionary of best matches into a sorted list of tuples.
    # Format: [(symptom_id, confidence), ...]
    final_results = sorted(best_matches.items(), key=lambda item: item[1], reverse=True)
    
    return final_results


# --- Test Program ---

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # --- NLP Component ---
    
    # Load the scispaCy model (used for sentence splitting).
    try:
        nlp = spacy.load("en_core_sci_lg")
        # print("scispaCy model 'en_core_sci_lg' loaded successfully.")
    except OSError:
        print("scispaCy model not found. Please run the installation commands again.")
        exit()

    print("\n--- Starting NLP Symptom Extraction Test ---")

    # Load the FINAL, most comprehensive knowledge base.
    try:
        with open("symptoms_final.json", "r") as f:
            symptoms_dict = json.load(f)
    except FileNotFoundError:
        print("Error: symptoms_final.json not found. Please ensure it is in the same directory.")
        exit()

    # The same 20 challenging test cases.
    # test_cases = [
    #     # Test Case 1: Classic Flu
    #     """I think I have the flu. I'm running a very high temperature and I can't stop shivering.
    #     My whole body is aching, especially my muscles, and I have a pounding headache.
    #     On top of that, I have a persistent cough and just feel completely worn out.""",
    #     # Test Case 2: Gastroenteritis (Stomach Bug)
    #     """I must have eaten something bad. I've been throwing up all night and have terrible the runs.
    #     My stomach is cramping really badly, I have no appetite at all, and I feel so queasy.
    #     I'm also worried I might be getting dehydrated from being sick so much.""",
    #     # Test Case 3: Urinary Tract Infection (UTI)
    #     """It's really painful when I go to the bathroom, like a burning sensation.
    #     I feel like I have to pee constantly, even when nothing comes out.
    #     I also noticed my urine is a very dark color and my lower back is aching.""",
    #     # Test Case 4: Allergic Reaction
    #     """I think I'm having an allergic reaction. I broke out in this awful skin rash all over my arms, and it's incredibly itchy.
    #     My nose has been running like a faucet and my eyes are watering constantly.
    #     I'm starting to feel a little short of breath too, which is scary.""",
    #     # Test Case 5: Migraine Attack
    #     """This is more than a normal headache, it's a full-blown migraine.
    #     The pain is throbbing behind my eyes and I'm so nauseous, I feel like I could vomit.
    #     My vision is also blurry and I'm seeing strange spots of light.""",
    #     # Test Case 6: Panic Attack
    #     """I had a scary episode where my heart started racing out of control.
    #     My chest felt incredibly tight, and I was overwhelmed with this feeling of dread and anxiety.
    #     I felt like I couldn't catch my breath no matter how hard I tried.""",
    #     # Test Case 7: Liver/Jaundice symptoms
    #     """Lately I've had zero energy, just a constant state of exhaustion.
    #     My partner noticed that my skin is starting to look yellowish, and the whites of my eyes are yellow too.
    #     I've also had this dull pain in my upper abdomen on the right side.""",
    #     # Test Case 8: Sinus Infection
    #     """I have so much sinus pressure in my face, it's giving me a bad head pain.
    #     My nose is completely stuffy and congested, and I have this thick phlegm in my chest.
    #     I also lost my sense of smell a couple of days ago.""",
    #     # Test Case 9: Joint Pain / Arthritis flare-up
    #     """My joints are so sore today, especially in my knees and hips.
    #     They feel swollen and it's making for very painful walking.
    #     There's also a lot of movement stiffness, particularly in the morning.""",
    #     # Test Case 10: Anemia / Low Iron
    #     """I've been feeling so tired and sluggish, with no energy for anything.
    #     I get dizzy and lightheaded if I stand up too fast, and my hands and feet are always cold.
    #     I also look very pale and sometimes I get palpitations.""",
    #     # Test Case 11: Dehydration and Malaise
    #     """I haven't been drinking enough water and I feel so unwell today.
    #     My mouth is dry, my head hurts, and I feel a general discomfort all over.
    #     I feel very weak and I'm not hungry at all.""",
    #     # Test Case 12: Neck and Back Strain
    #     """I think I slept wrong because I woke up with a very stiff neck and a sore back.
    #     The pain in my neck makes it hard to turn my head.
    #     My back muscles feel like they are in a spasm.""",
    #     # Test Case 13: Skin infection
    #     """I have this red sore around my nose that's very tender.
    #     It's developed a yellow crust that oozes a little bit.
    #     There are also some pus-filled pimples nearby.""",
    #     # Test Case 14: Confusion and Weakness (Serious)
    #     """I'm worried about my dad. He seems very confused and disoriented today.
    #     He's complaining of weakness on one side of his body and has slurred speech.
    #     He also says he's seeing things strangely, like visual disturbances.""",
    #     # Test Case 15: Heart-related concern
    #     """I've been having these episodes where my heart feels like it's fluttering in my chest.
    #     It's not exactly a pain, more like a rapid heartbeat that feels irregular.
    #     It makes me feel a bit faint and anxious when it happens.""",
    #     # Test Case 16: Bad Cold
    #     """I have a classic head cold. It started with a scratchy throat and now I can't stop sneezing.
    #     My nose is running constantly and I have a hacking cough that keeps me up at night.
    #     I just feel generally unwell.""",
    #     # Test Case 17: Digestive Issues
    #     """My stomach has been bloated for days, and I'm very gassy.
    #     I'm constipated and it's a painful bowel movement when I can go.
    #     I'm also getting a lot of acid reflux after eating.""",
    #     # Test Case 18: Eye Irritation
    #     """My eyes are red and have been watering non-stop.
    #     There's a bit of an itchy feeling, almost like an internal itching.
    #     My vision seems a bit blurry because of all the wateriness.""",
    #     # Test Case 19: Unexplained Weight Loss
    #     """I've been losing weight without trying to, which is concerning.
    #     I've had a total loss of appetite and feel exhausted all the time.
    #     I've also been sweating a lot more than usual, even when it's not hot.""",
    #     # Test Case 20: General Viral Symptoms
    #     """I started feeling unwell with a low-grade fever and some body aches.
    #     Now I have a sore throat and feel a lot of fatigue.
    #     It's just a feeling of general discomfort, I can't put my finger on it.""",
    # ]

    test_cases = ["I am feeling fever and pain in my throat. i also have runny nose and difficulty breathing."]

    # Run each test case through the extraction function
    for i, text in enumerate(test_cases):
        print("-" * 50)
        print(f"Test Case #{i+1}")
        print(f"Input Text: \"\"\"{text}\"\"\"")

        # The function now returns a list of (id, confidence) tuples
        extracted_results = extract_symptoms_from_text(text, symptoms_dict, nlp)
        
        print(f"Extracted Symptoms ({len(extracted_results)} found):")
        if not extracted_results:
            print("  None")
        else:
            for symptom_id, confidence in extracted_results:
                name = symptoms_dict.get(symptom_id)
                primary_name = name[0] if isinstance(name, list) else name
                print(f"  - {primary_name:<30} (Confidence: {confidence}%) -> ID: {symptom_id}")
        print()

    print("--- Test Complete ---")