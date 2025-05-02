'''
cs5322s25.py

Module for Word Sense Disambiguation (WSD) for words:
- camper
- conviction
- deed

Each WSD_Test_* function loads its pre-trained model and vectorizer, then predicts sense 1 or 2 for each input sentence.

train_camper() reads prog3/camper.txt where lines follow:
1) word
2) "1   gloss1"
3) "2   gloss2"
Then each subsequent line is "1 sentence..." or "2 sentence...".
'''
import os
import re
import joblib
import numpy as np
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')

from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords

# Base directory of this module
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Paths for 'camper'
CAMPER_MODEL_PATH = os.path.join(MODELS_DIR, 'camper_model.pkl')
CAMPER_VECTORIZER_PATH = os.path.join(MODELS_DIR, 'camper_vectorizer.pkl')

# Paths for 'conviction'
CONV_MODEL_PATH = os.path.join(MODELS_DIR, 'conviction_model.pkl')
CONV_VECTORIZER_PATH = os.path.join(MODELS_DIR, 'conviction_vectorizer.pkl')

# Paths for 'deed'
DEED_MODEL_PATH = os.path.join(MODELS_DIR, 'deed_model.pkl')
DEED_VECTORIZER_PATH = os.path.join(MODELS_DIR, 'deed_vectorizer.pkl')

# Legal terms for conviction sense disambiguation
LEGAL_TERMS = {
    'court', 'trial', 'judge', 'jury', 'sentence', 'criminal', 'appeal', 'guilty', 'charged',
    'prosecution', 'defendant', 'attorney', 'evidence', 'verdict', 'plea', 'felony', 'jail',
    'prison', 'arrest', 'law', 'legal', 'crime', 'offense', 'punishment', 'convicted',
    # Additional legal context terms
    'charges', 'case', 'ruling', 'justice', 'court', 'courts', 'criminal', 'crimes',
    'sentenced', 'sentenced', 'pleaded', 'plead', 'guilty', 'acquittal', 'appeals',
    'prosecutor', 'prosecutors', 'defense', 'judge', 'judges', 'judicial', 'testimony',
    'witness', 'witnesses', 'trial', 'trials', 'verdict', 'verdicts', 'sentenced',
    'sentencing', 'prison', 'jail', 'incarceration', 'parole', 'probation', 'bail',
    'arrest', 'arrested', 'charges', 'charged', 'indictment', 'indicted', 'allegations',
    'alleged', 'defendant', 'defendants', 'prosecution', 'prosecutor', 'prosecutors',
    'plea', 'pleas', 'pleading', 'appeal', 'appeals', 'appealing', 'appealed'
}

# Non-legal context indicators
BELIEF_TERMS = {
    'belief', 'believe', 'believed', 'believing', 'faith', 'confident', 'confidence',
    'sure', 'certain', 'certainty', 'strong', 'strongly', 'firm', 'firmly', 'deep',
    'deeply', 'passionate', 'passionately', 'absolute', 'absolutely', 'complete',
    'completely', 'total', 'totally', 'utter', 'utterly', 'sincere', 'sincerely',
    'sincerity', 'genuine', 'genuinely', 'authentic', 'authentically', 'true', 'truly',
    'real', 'really', 'honest', 'honestly', 'heartfelt', 'personal', 'personally',
    'moral', 'morally', 'ethical', 'ethically', 'principle', 'principled', 'belief',
    'beliefs', 'believe', 'believes', 'believed', 'believing', 'believer', 'believers'
}

# Vehicle/Structure related terms for camper sense 1
VEHICLE_TERMS = {
    'van', 'rv', 'vehicle', 'truck', 'trailer', 'wheel', 'motor', 'drive', 'drove', 'park',
    'parked', 'parking', 'sleep', 'sleeping', 'slept', 'inside', 'car', 'boat', 'road',
    'trip', 'travel', 'traveling', 'window', 'door', 'roof', 'tire', 'engine', 'seat',
    'seats', 'driver', 'passenger', 'gear', 'brake', 'gas', 'fuel', 'highway', 'road',
    'miles', 'mileage', 'garage', 'storage', 'stored', 'store', 'equipment', 'remodel',
    'repair', 'fix', 'install', 'installation', 'upgrade', 'maintenance', 'replace',
    'replacement', 'part', 'parts', 'accessory', 'accessories', 'recreational', 'vehicle',
    'pop-up', 'popup', 'fifth-wheel', 'fifth', 'wheel', 'motorhome', 'motor', 'home'
}

# Person/Camping activity terms for camper sense 2
CAMPING_TERMS = {
    'camp', 'camping', 'hike', 'hiking', 'outdoor', 'outdoors', 'trail', 'trails',
    'wilderness', 'woods', 'forest', 'mountain', 'mountains', 'backpack', 'backpacking',
    'tent', 'tents', 'gear', 'equipment', 'adventure', 'adventurer', 'explore', 'explorer',
    'exploring', 'nature', 'natural', 'wild', 'wildlife', 'park', 'parks', 'national',
    'state', 'recreation', 'recreational', 'activity', 'activities', 'experience',
    'experienced', 'skill', 'skills', 'skilled', 'expert', 'novice', 'beginner',
    'advanced', 'professional', 'amateur', 'enthusiast', 'lover', 'passionate',
    'counselor', 'guide', 'instructor', 'leader', 'participant', 'member', 'group',
    'team', 'crew', 'staff', 'volunteer', 'program', 'camp', 'camps', 'campsite',
    'campsites', 'campground', 'campgrounds', 'summer', 'season', 'seasonal'
}

# Add deed-specific term sets
LEGAL_DOC_TERMS = {
    'deed', 'property', 'legal', 'document', 'signed', 'seal', 'transfer', 'ownership',
    'title', 'contract', 'agreement', 'covenant', 'estate', 'mortgage', 'conveyance',
    'real estate', 'property rights', 'land', 'house', 'home', 'building', 'lawyer',
    'attorney', 'notary', 'signature', 'sign', 'signed', 'signing', 'official',
    'documentation', 'paperwork', 'certificate', 'certified', 'record', 'recorded',
    'registry', 'registrar', 'courthouse', 'county', 'legal rights', 'ownership rights',
    'property owner', 'buyer', 'seller', 'sale', 'purchase', 'transaction'
}

ACTION_TERMS = {
    'act', 'action', 'deed', 'accomplish', 'achievement', 'perform', 'done', 'doing',
    'help', 'service', 'volunteer', 'charity', 'charitable', 'kindness', 'kind',
    'good', 'heroic', 'brave', 'courage', 'honor', 'honorable', 'noble', 'selfless',
    'generous', 'generosity', 'sacrifice', 'sacrificial', 'effort', 'contribution',
    'impact', 'difference', 'change', 'positive', 'meaningful', 'significant',
    'important', 'memorable', 'unforgettable', 'remarkable', 'outstanding'
}

def preprocess_text(text: str) -> dict:
    """
    Preprocess text with key features for WSD.
    """
    # Tokenize
    text_lower = text.lower()
    tokens = text_lower.split()
    
    # Get context around target word
    try:
        target_idx = tokens.index('conviction')
        context_before = ' '.join(tokens[max(0, target_idx-5):target_idx])
        context_after = ' '.join(tokens[target_idx+1:target_idx+6])
    except ValueError:
        context_before = ''
        context_after = ''
    
    # Count terms
    legal_term_count = sum(1 for token in tokens if token in LEGAL_TERMS)
    belief_term_count = sum(1 for token in tokens if token in BELIEF_TERMS)
    
    # Check for legal phrases
    legal_phrases = [
        'found guilty', 'criminal case', 'court ruling', 'legal proceeding',
        'criminal charge', 'criminal record', 'court case', 'legal case',
        'criminal conviction', 'prior conviction', 'felony conviction',
        'guilty plea', 'court appeal', 'appeals court', 'supreme court',
        'district court', 'federal court', 'state court', 'criminal court'
    ]
    legal_phrase_count = sum(1 for phrase in legal_phrases if phrase in text_lower)
    
    return {
        'text': text,
        'context_before': context_before,
        'context_after': context_after,
        'legal_term_count': legal_term_count,
        'belief_term_count': belief_term_count,
        'legal_phrase_count': legal_phrase_count
    }

def preprocess_camper_text(text: str) -> dict:
    """
    Preprocess text with key features for camper WSD.
    """
    # Tokenize
    text_lower = text.lower()
    tokens = text_lower.split()
    
    # Get context around target word
    try:
        target_idx = tokens.index('camper')
        if target_idx == -1:  # Try plural form
            target_idx = tokens.index('campers')
        context_before = ' '.join(tokens[max(0, target_idx-5):target_idx])
        context_after = ' '.join(tokens[target_idx+1:target_idx+6])
    except ValueError:
        context_before = ''
        context_after = ''
    
    # Count terms
    vehicle_term_count = sum(1 for token in tokens if token in VEHICLE_TERMS)
    camping_term_count = sum(1 for token in tokens if token in CAMPING_TERMS)
    
    # Check for vehicle/structure phrases
    vehicle_phrases = [
        'pop up', 'fifth wheel', 'motor home', 'recreational vehicle', 'rv', 
        'van', 'truck', 'trailer', 'mobile home', 'park the', 'parked the',
        'inside the', 'drive the', 'drove the', 'sleep in', 'sleeping in',
        'slept in', 'living in', 'lived in', 'stay in', 'stayed in'
    ]
    
    # Check for camping/person phrases
    camping_phrases = [
        'experienced', 'seasoned', 'avid', 'each', 'every', 'the', 'a',
        'camp', 'camping', 'hiker', 'hiking', 'outdoor', 'outdoors',
        'backpack', 'backpacking', 'adventure', 'adventurous'
    ]
    
    vehicle_phrase_count = sum(1 for phrase in vehicle_phrases if phrase in text_lower)
    camping_phrase_count = sum(1 for phrase in camping_phrases if phrase in text_lower)
    
    # Check for determiners before camper that suggest person
    person_indicators = ['the', 'a', 'an', 'each', 'every', 'this', 'that']
    has_person_determiner = any(
        det + ' camper' in text_lower for det in person_indicators
    )
    
    return {
        'text': text,
        'context_before': context_before,
        'context_after': context_after,
        'vehicle_term_count': vehicle_term_count,
        'camping_term_count': camping_term_count,
        'vehicle_phrase_count': vehicle_phrase_count,
        'camping_phrase_count': camping_phrase_count,
        'has_person_determiner': int(has_person_determiner)
    }

def _load_model_and_vectorizer(model_path: str, vectorizer_path: str):
    """
    Load a serialized model and its vectorizer from disk.
    Returns:
        model: the trained classifier
        vectorizer: the fitted vectorizer
    """
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def WSD_Test_camper(sentences: List[str]) -> List[int]:
    """
    Improved WSD for 'camper' using enhanced features.
    """
    model, (vectorizer, text_feat_size) = _load_model_and_vectorizer(CAMPER_MODEL_PATH, CAMPER_VECTORIZER_PATH)
    
    # Preprocess all sentences
    processed = [preprocess_camper_text(sent) for sent in sentences]
    
    # Extract features
    X = vectorizer.transform([t['text'] for t in processed])
    additional_features = np.array([
        [t['vehicle_term_count'], t['camping_term_count'],
         t['vehicle_phrase_count'], t['camping_phrase_count'],
         t['has_person_determiner']]
        for t in processed
    ])
    X_with_features = np.hstack([X.toarray(), additional_features])
    
    # Predict
    predictions = model.predict(X_with_features)
    return [int(p) for p in predictions]

def train_conviction_model():
    """
    Train an improved WSD model for 'conviction' using enhanced features.
    """
    data_file = os.path.join(BASE_DIR, 'prog3', 'conviction.txt')
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    # Read and parse training data
    with open(data_file, encoding='utf-8') as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    
    # Skip header lines and process only valid training lines
    training_lines = []
    for line in lines:
        if line[0] in ('1', '2') and ' ' in line:
            # Ensure we only get lines that start with 1 or 2 and have text
            training_lines.append(line)
    
    # Prepare training data
    texts = []
    labels = []
    for line in training_lines:
        parts = line.split(' ', 1)
        if len(parts) == 2:
            label, text = parts
            if label in ('1', '2'):  # Extra validation
                texts.append(text)
                labels.append(int(label))
    
    if not texts:
        raise ValueError("No valid training examples found in conviction.txt")
    
    # Preprocess all texts
    processed_texts = [preprocess_text(text) for text in texts]
    
    # Create feature extraction pipeline
    text_features = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=1000,
        stop_words='english'
    )
    
    # Train model
    X = text_features.fit_transform([t['text'] for t in processed_texts])
    additional_features = np.array([
        [t['legal_term_count'], t['belief_term_count'], t['legal_phrase_count']]
        for t in processed_texts
    ])
    X_with_features = np.hstack([X.toarray(), additional_features])
    
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_with_features, labels)
    
    # Save model and vectorizer
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, CONV_MODEL_PATH)
    joblib.dump((text_features, X.shape[1]), CONV_VECTORIZER_PATH)
    print(f"Trained conviction model on {len(texts)} sentences and saved to '{MODELS_DIR}'")

def WSD_Test_conviction(sentences: List[str]) -> List[int]:
    """
    Improved WSD for 'conviction' using enhanced features.
    """
    model, (vectorizer, text_feat_size) = _load_model_and_vectorizer(CONV_MODEL_PATH, CONV_VECTORIZER_PATH)
    
    # Preprocess all sentences
    processed = [preprocess_text(sent) for sent in sentences]
    
    # Extract features
    X = vectorizer.transform([t['text'] for t in processed])
    additional_features = np.array([
        [t['legal_term_count'], t['belief_term_count'], t['legal_phrase_count']]
        for t in processed
    ])
    X_with_features = np.hstack([X.toarray(), additional_features])
    
    # Predict
    predictions = model.predict(X_with_features)
    return [int(p) for p in predictions]

def preprocess_deed_text(text: str) -> dict:
    """Preprocess text for deed sense disambiguation with custom features."""
    # Basic preprocessing
    text = text.lower()
    
    # Count legal document terms
    legal_term_count = sum(1 for term in LEGAL_DOC_TERMS if term in text)
    
    # Count action/achievement terms
    action_term_count = sum(1 for term in ACTION_TERMS if term in text)
    
    # Count legal document phrases
    legal_phrases = [
        'sign the deed', 'property deed', 'deed of', 'transfer deed',
        'deed to the', 'legal document', 'official document', 'property rights'
    ]
    legal_phrase_count = sum(1 for phrase in legal_phrases if phrase in text)
    
    # Count action phrases
    action_phrases = [
        'good deed', 'brave deed', 'heroic deed', 'kind deed',
        'deed of kindness', 'charitable deed', 'noble deed'
    ]
    action_phrase_count = sum(1 for phrase in action_phrases if phrase in text)
    
    return {
        'text': text,
        'legal_term_count': legal_term_count,
        'action_term_count': action_term_count,
        'legal_phrase_count': legal_phrase_count,
        'action_phrase_count': action_phrase_count
    }

def train_deed_model():
    """Train and save the deed model and vectorizer."""
    data_file = os.path.join(BASE_DIR, 'prog3', 'deed.txt')
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    # Read and parse training data
    with open(data_file, encoding='utf-8') as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    
    # Skip header lines (word and glosses)
    training_lines = []
    for line in lines[3:]:  # Skip first 3 lines
        if line[0] in ('1', '2') and ' ' in line:
            # Handle labels with periods (e.g., "1." or "2.")
            label = line.split(' ', 1)[0].rstrip('.')
            if label in ('1', '2'):
                training_lines.append(line)
    
    # Prepare training data
    texts = []
    labels = []
    for line in training_lines:
        label = line.split(' ', 1)[0].rstrip('.')
        text = line.split(' ', 1)[1]
        texts.append(text)
        labels.append(int(label))
    
    if not texts:
        raise ValueError("No valid training examples found in deed.txt")
    
    # Preprocess all texts
    processed_texts = [preprocess_deed_text(text) for text in texts]
    
    # Create feature extraction pipeline
    text_features = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=1000,
        stop_words='english'
    )
    
    # Train model
    X = text_features.fit_transform([t['text'] for t in processed_texts])
    additional_features = np.array([
        [t['legal_term_count'], t['action_term_count'],
         t['legal_phrase_count'], t['action_phrase_count']]
        for t in processed_texts
    ])
    X_with_features = np.hstack([X.toarray(), additional_features])
    
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_with_features, labels)
    
    # Save model and vectorizer
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, DEED_MODEL_PATH)
    joblib.dump((text_features, X.shape[1]), DEED_VECTORIZER_PATH)
    print(f"Trained deed model on {len(texts)} sentences and saved to '{MODELS_DIR}'")

def WSD_Test_deed(sentences: List[str]) -> List[int]:
    """
    Improved WSD for 'deed' using enhanced features.
    """
    model, (vectorizer, text_feat_size) = _load_model_and_vectorizer(DEED_MODEL_PATH, DEED_VECTORIZER_PATH)
    
    # Preprocess all sentences
    processed = [preprocess_deed_text(sent) for sent in sentences]
    
    # Extract features
    X = vectorizer.transform([t['text'] for t in processed])
    additional_features = np.array([
        [t['legal_term_count'], t['action_term_count'],
         t['legal_phrase_count'], t['action_phrase_count']]
        for t in processed
    ])
    X_with_features = np.hstack([X.toarray(), additional_features])
    
    # Predict
    predictions = model.predict(X_with_features)
    return [int(p) for p in predictions]

def train_camper():
    """
    Train an improved WSD model for 'camper' using enhanced features.
    """
    data_file = os.path.join(BASE_DIR, 'prog3', 'camper.txt')
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    # Read and parse training data
    with open(data_file, encoding='utf-8') as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    
    # Skip header lines (word and glosses)
    training_lines = []
    for line in lines[3:]:  # Skip first 3 lines
        if line[0] in ('1', '2') and ' ' in line:
            training_lines.append(line)
    
    # Prepare training data
    texts = []
    labels = []
    for line in training_lines:
        parts = line.split(' ', 1)
        if len(parts) == 2:
            label, text = parts
            if label in ('1', '2'):
                texts.append(text)
                labels.append(int(label))
    
    if not texts:
        raise ValueError("No valid training examples found in camper.txt")
    
    # Preprocess all texts
    processed_texts = [preprocess_camper_text(text) for text in texts]
    
    # Create feature extraction pipeline
    text_features = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=1000,
        stop_words='english'
    )
    
    # Train model
    X = text_features.fit_transform([t['text'] for t in processed_texts])
    additional_features = np.array([
        [t['vehicle_term_count'], t['camping_term_count'],
         t['vehicle_phrase_count'], t['camping_phrase_count'],
         t['has_person_determiner']]
        for t in processed_texts
    ])
    X_with_features = np.hstack([X.toarray(), additional_features])
    
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_with_features, labels)
    
    # Save model and vectorizer
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, CAMPER_MODEL_PATH)
    joblib.dump((text_features, X.shape[1]), CAMPER_VECTORIZER_PATH)
    print(f"Trained camper model on {len(texts)} sentences and saved to '{MODELS_DIR}'")

if __name__ == '__main__':
    train_conviction_model()
    train_camper()
    train_deed_model()
