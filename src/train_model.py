import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from gensim.models import Word2Vec
from preprocess import preprocess_text

df = pd.read_csv("data/dataset.csv")
df["Text"] = df["Text"].apply(preprocess_text)

vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(df["Text"])
y = df["Label"]

smote = SMOTE(random_state=42)
X_tfidf_resampled, y_resampled = smote.fit_resample(X_tfidf, y)

clf_tfidf = MultinomialNB()
clf_tfidf.fit(X_tfidf_resampled, y_resampled)

sentences = [text.split() for text in df["Text"]]
word2vec_model = Word2Vec(sentences, vector_size=300, min_count=1, workers=4)

X_word2vec = np.array([
    np.mean([word2vec_model.wv[word] for word in text if word in word2vec_model.wv] or [np.zeros(300)], axis=0)
    for text in sentences
])

X_word2vec_resampled, y_resampled_w2v = smote.fit_resample(X_word2vec, y)

clf_word2vec = RandomForestClassifier(n_estimators=100, random_state=42)
clf_word2vec.fit(X_word2vec_resampled, y_resampled_w2v)

joblib.dump(vectorizer, "../models/tfidf_vectorizer.pkl")
joblib.dump(clf_tfidf, "../models/fraud_classifier_tfidf.pkl")
word2vec_model.save("../models/word2vec_model.bin")
joblib.dump(clf_word2vec, "../models/fraud_classifier_word2vec.pkl")
