from thefuzz import fuzz
import re

CONFIDENCE_THRESHOLD = 85

# Comprehensive list of negation words and patterns
NEGATION_WORDS = {
    'no', 'not', 'never', 'neither', 'nor', 'none', 'nobody', 'nothing', 'nowhere',
    "don't", "doesn't", "didn't", "won't", "wouldn't", "shouldn't", "couldn't", 
    "can't", "cannot", "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't",
    "hadn't", "without", 'lack', 'lacking', 'absent', 'free', 'minus', 'except'
}

def contains_negation(sentence, symptom_phrase, window_size=5):
    """
    Checks if a sentence contains negation words near the symptom phrase.
    """
    sentence_lower = sentence.lower()
    symptom_lower = symptom_phrase.lower()
    
    # Tokenize the sentence into words (preserve contractions like "don't")
    words = re.findall(r"\b[\w']+\b", sentence_lower)
    symptom_words = set(re.findall(r"\b[\w']+\b", symptom_lower))
    
    # If we can't find the symptom words in the sentence, no negation check needed
    if not symptom_words:
        return False
    
    # Find positions where ANY symptom words appear
    symptom_positions = []
    for i, word in enumerate(words):
        # Check if this word is part of the symptom phrase
        if word in symptom_words:
            symptom_positions.append(i)
    
    # If no symptom words found, no negation
    if not symptom_positions:
        return False
    
    # Check for negation words before the FIRST occurrence of any symptom word
    first_symptom_pos = min(symptom_positions)
    start_idx = max(0, first_symptom_pos - window_size)
    window = words[start_idx:first_symptom_pos]
    
    # Check if any negation word appears in the window
    for word in window:
        if word in NEGATION_WORDS:
            return True
    
    return False

def extract_symptoms_from_text(text, symptom_dict, nlp_model):
    """
    Extracts symptoms using robust fuzzy string matching with negation detection,
    returning only high-confidence results that are not negated.
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
            # Compare each synonym against each user sentence.
            for sentence in user_sentences:
                # Calculate the confidence score using a robust fuzzy matching algorithm.
                # token_set_ratio is excellent as it ignores word order and extra words.
                confidence = fuzz.token_set_ratio(synonym.lower(), sentence.lower())

                # If the confidence is high enough, consider it a potential match.
                if confidence >= CONFIDENCE_THRESHOLD:
                    # Check for negation - skip this match if negation is detected
                    if contains_negation(sentence, synonym):
                        continue
                    
                    # Check if this symptom has been found before with a lower confidence.
                    # If this match is better, update it.
                    if symptom_id not in best_matches or confidence > best_matches[symptom_id]:
                        best_matches[symptom_id] = confidence

    # Convert the dictionary of best matches into a sorted list of tuples.
    final_results = sorted(best_matches.items(), key=lambda item: item[1], reverse=True)
    
    return final_results