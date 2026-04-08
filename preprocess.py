import re
from xml.parsers.expat import model
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('omw-1.4', quiet=True)

def preprocess(q):
    q = str(q).lower().strip()
    q = BeautifulSoup(q, "html.parser").get_text()
    q = q.replace('%', ' percent ')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')
    q = q.replace('[math]', '')


    q = q.replace(',', '') 
    q = re.sub(r'(\d+)000000000', r'\1b', q)
    q = re.sub(r'(\d+)000000', r'\1m', q)
    q = re.sub(r'(\d+)000', r'\1k', q)
    
    contractions = { 
        "ain't": "am not", "aren't": "are not", "can't": "can not", "can't've": "can not have",
        "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",
        "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not",
        "haven't": "have not", "he'd": "he would", "he'll": "he will", "he's": "he is",
        "how'd": "how did", "how'll": "how will", "how's": "how is", "i'd": "i would",
        "i'll": "i will", "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
        "it'll": "it will", "it's": "it is", "let's": "let us", "must've": "must have",
        "mustn't": "must not", "shan't": "shall not", "she'd": "she would", "she'll": "she will",
        "she's": "she is", "should've": "should have", "shouldn't": "should not", "that's": "that is",
        "there's": "there is", "they'd": "they would", "they'll": "they will", "they're": "they are",
        "they've": "they have", "wasn't": "was not", "we'd": "we would", "we'll": "we will",
        "we're": "we are", "we've": "we have", "weren't": "were not", "what's": "what is",
        "where's": "where is", "who's": "who is", "won't": "will not", "would've": "would have",
        "wouldn't": "would not", "you'd": "you would", "you'll": "you will", "you're": "you are",
        "you've": "you have"
    }
    
    q_decontracted = [contractions[word] if word in contractions else word for word in q.split()]
    q = ' '.join(q_decontracted)
    

    q = re.sub(r'[^a-zA-Z0-9\s]', ' ', q)
    stop_words = set(stopwords.words('english'))
    keep_list = {'how', 'why', 'what', 'when', 'where', 'not', 'no', 'nor', 'neither', 'none'}
    final_stop_words = stop_words - keep_list
    
    lemmatizer = WordNetLemmatizer()
    words = q.split()
    q = " ".join([lemmatizer.lemmatize(word) for word in words if word not in final_stop_words])
    
    return q.strip()