from preprocess import preprocess
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import joblib
import os

base_path = os.path.dirname(__file__)
freq_path = os.path.join(base_path, "Model", "freq_map.pkl")

GLOBAL_FREQ_MAP = joblib.load("Model/freq_map.pkl")
    
model = SentenceTransformer('all-MiniLM-L6-v2')

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import paired_cosine_distances

def common_words(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
    return len(w1 & w2)

def total_words(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
    return (len(w1) + len(w2))

def fetch_token_features(row):
    
    q1 = row['question1']
    q2 = row['question2']
    
    SAFE_DIV = 0.0001 

    STOP_WORDS = stopwords.words("english")
    
    token_features = [0.0]*8
    
    # Converting the Sentence into Tokens: 
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    # Get the non-stopwords in Questions
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])

    #Get the stopwords in Questions
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])
    
    # Get the common non-stopwords from Question pair
    common_word_count = len(q1_words.intersection(q2_words))
    
    # Get the common stopwords from Question pair
    common_stop_count = len(q1_stops.intersection(q2_stops))
    
    # Get the common Tokens from Question pair
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
    
    
    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    
    # Last word of both question is same or not
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])

    # First word of both question is same or not
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    
    return token_features


import distance

def fetch_length_features(row):
    
    q1 = row['question1']
    q2 = row['question2']
    
    length_features = [0.0]*3
    
    # Converting the Sentence into Tokens: 
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return length_features
    
    # Absolute length features
    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
    
    #Average Token Length of both Questions
    length_features[1] = (len(q1_tokens) + len(q2_tokens))/2
    
    strs = list(distance.lcsubstrings(q1, q2))
    if strs:
        length_features[2] = len(strs[0]) / (min(len(q1), len(q2)) + 1)
    else:
        length_features[2] = 0.0
    
    return length_features

from fuzzywuzzy import fuzz

def fetch_fuzzy_features(row):
    
    q1 = row['question1']
    q2 = row['question2']
    
    fuzzy_features = [0.0]*4
    
    # fuzz_ratio
    fuzzy_features[0] = fuzz.QRatio(q1, q2)

    # fuzz_partial_ratio
    fuzzy_features[1] = fuzz.partial_ratio(q1, q2)

    # token_sort_ratio
    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)
    fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)

    return fuzzy_features

import pandas as pd
import spacy

# 1. Load model with only what we need
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def get_nlp_ratios(df):
    # Process all questions in large batches
    q1_docs = list(nlp.pipe(df['question1'].astype(str), batch_size=1000))
    q2_docs = list(nlp.pipe(df['question2'].astype(str), batch_size=1000))
    
    noun_ratios = []
    verb_ratios = []
    
    for doc1, doc2 in zip(q1_docs, q2_docs):
        # Extract sets of nouns and verbs
        n1 = {token.text.lower() for token in doc1 if token.pos_ == 'NOUN'}
        n2 = {token.text.lower() for token in doc2 if token.pos_ == 'NOUN'}
        v1 = {token.text.lower() for token in doc1 if token.pos_ == 'VERB'}
        v2 = {token.text.lower() for token in doc2 if token.pos_ == 'VERB'}
        
        # Calculate Jaccard-style ratios
        n_union = len(n1 | n2)
        v_union = len(v1 | v2)
        
        noun_ratios.append(len(n1 & n2) / n_union if n_union > 0 else 0)
        verb_ratios.append(len(v1 & v2) / v_union if v_union > 0 else 0)
        
    return noun_ratios, verb_ratios

def get_ner_batch(questions):
    # Disable components we don't need to save speed
    with nlp.select_pipes(enable=['tok2vec', 'ner']):
        docs = list(nlp.pipe(questions, batch_size=1000))
    return [set([ent.text for ent in doc.ents]) for doc in docs]

def fast_jaccard(df):
    results = []
    # Using zip is significantly faster than .apply(axis=1)
    for q1, q2 in zip(df['question1'], df['question2']):
        s1 = set(str(q1).lower().split())
        s2 = set(str(q2).lower().split())
        inter = len(s1 & s2)
        union = len(s1 | s2)
        results.append(inter / union if union != 0 else 0)
    return results

# 2. Compare the sets
def compare_ents(e1, e2):
    if len(e1) == 0 and len(e2) == 0: return 1.0
    return 1.0 if len(e1.intersection(e2)) > 0 else 0.0

def create_features(q1, q2):
    q1 = preprocess(q1)
    q2 = preprocess(q2)
    features = {}
    features['question1'] = q1
    features['question2'] = q2
    #Basic features
    features['q1_len'] = len(features['question1'])
    features['q2_len'] = len(features['question2'])
    features['q1_num_words'] = len(features['question1'].split(" "))
    features['q2_num_words'] = len(features['question2'].split(" "))

    df = pd.DataFrame([features])
    df['word_common'] = df.apply(common_words, axis=1)
    df['word_total'] = df.apply(total_words, axis=1)
    df['word_share'] = round(df['word_common']/df['word_total'], 2)

    token_features = df.apply(fetch_token_features, axis=1)

    df["cwc_min"]       = list(map(lambda x: x[0], token_features))
    df["cwc_max"]       = list(map(lambda x: x[1], token_features))
    df["csc_min"]       = list(map(lambda x: x[2], token_features))
    df["csc_max"]       = list(map(lambda x: x[3], token_features))
    df["ctc_min"]       = list(map(lambda x: x[4], token_features))
    df["ctc_max"]       = list(map(lambda x: x[5], token_features))
    df["last_word_eq"]  = list(map(lambda x: x[6], token_features))
    df["first_word_eq"] = list(map(lambda x: x[7], token_features))

    length_features = df.apply(fetch_length_features, axis=1)

    df['abs_len_diff'] = list(map(lambda x: x[0], length_features))
    df['mean_len'] = list(map(lambda x: x[1], length_features))
    df['longest_substr_ratio'] = list(map(lambda x: x[2], length_features))
    
    fuzzy_features = df.apply(fetch_fuzzy_features, axis=1)
    # Creating new feature columns for fuzzy features
    df['fuzz_ratio'] = list(map(lambda x: x[0], fuzzy_features))
    df['fuzz_partial_ratio'] = list(map(lambda x: x[1], fuzzy_features))
    df['token_sort_ratio'] = list(map(lambda x: x[2], fuzzy_features))
    df['token_set_ratio'] = list(map(lambda x: x[3], fuzzy_features))

    
    df['q1_freq'] = df['question1'].map(lambda x: GLOBAL_FREQ_MAP.get(str(x), 1))
    df['q2_freq'] = df['question2'].map(lambda x: GLOBAL_FREQ_MAP.get(str(x), 1))

    df['noun_ratio'], df['verb_ratio'] = get_nlp_ratios(df)

    q1_embeddings = model.encode(df['question1'].tolist(), show_progress_bar=True)
    q2_embeddings = model.encode(df['question2'].tolist(), show_progress_bar=True)
    df['cosine_sim'] = 1 - paired_cosine_distances(q1_embeddings, q2_embeddings)

    q1_ents = get_ner_batch(df['question1'].astype(str))
    q2_ents = get_ner_batch(df['question2'].astype(str))
    df['ner_overlap'] = [compare_ents(a, b) for a, b in zip(q1_ents, q2_ents)]
    df['jaccard_sim'] = fast_jaccard(df)

    return df

