import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text) 
    text = re.sub(r'\W', ' ', text) 
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)
