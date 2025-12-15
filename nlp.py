from thefuzz import fuzz

CONFIDENCE_THRESHOLD = 85

def extract_symptoms_from_text(text, symptom_dict, nlp_model):
    """
    Extracts symptoms using robust fuzzy string matching, returning only high-confidence results.
    """
    # Use a dictionary to store the best match for each symptom to avoid duplicates.
    best_matches = {}

    # Split the user's text into individual sentences.
    doc = nlp_model(text)
    user_sentences = [sent.text for sent in doc.sents]

    # Iterate through every symptom and its known synonyms.
    for symptom_id, value in symptom_dict.items():
        synonym_phrases = value if isinstance(value, list) else [value]
        
        for synonym in synonym_phrases:
            # 3. Compare each synonym against each user sentence.
            for sentence in user_sentences:
                # Calculate the confidence score using a robust fuzzy matching algorithm.
                # token_set_ratio is excellent as it ignores word order and extra words.
                confidence = fuzz.token_set_ratio(synonym.lower(), sentence.lower())

                # If the confidence is high enough, consider it a potential match.
                if confidence >= CONFIDENCE_THRESHOLD:
                    # Check if this symptom has been found before with a lower confidence.
                    # If this match is better, update it.
                    if symptom_id not in best_matches or confidence > best_matches[symptom_id]:
                        best_matches[symptom_id] = confidence

    # Convert the dictionary of best matches into a sorted list of tuples.
    final_results = sorted(best_matches.items(), key=lambda item: item[1], reverse=True)
    
    return final_results