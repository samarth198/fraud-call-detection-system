import time
import numpy as np
import joblib
import speech_recognition as sr
from gensim.models import Word2Vec
from preprocess import preprocess_text

vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")
clf_tfidf = joblib.load("../models/fraud_classifier_tfidf.pkl")
word2vec_model = Word2Vec.load("../models/word2vec_model.bin")
clf_word2vec = joblib.load("../models/fraud_classifier_word2vec.pkl")

def hybrid_prediction(text):
    processed_text = preprocess_text(text)

    tfidf_vector = vectorizer.transform([processed_text])
    tfidf_pred = clf_tfidf.predict(tfidf_vector)[0]

    word_vectors = [word2vec_model.wv[word] for word in processed_text.split() if word in word2vec_model.wv]
    word2vec_vector = np.mean(word_vectors or [np.zeros(300)], axis=0).reshape(1, -1)
    word2vec_pred = clf_word2vec.predict(word2vec_vector)[0]

    return "Fraud" if tfidf_pred == "fraud" and word2vec_pred == "fraud" else "Normal"

def listen_live_audio():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("Listening for 10 seconds...")
        recognizer.adjust_for_ambient_noise(source)
        audio_data = recognizer.record(source, duration=10)

    try:
        text = recognizer.recognize_google(audio_data)
        print(f"Recognized Text: {text}")
        prediction = hybrid_prediction(text)
        print(f"Live Call Prediction: {prediction}")
        return text, prediction
    except sr.UnknownValueError:
        print("Speech recognition could not understand audio.")
        return None, None
    except sr.RequestError:
        print("Error in API request.")
        return None, None

if __name__ == "__main__":
    print("🔴 Live Fraud Detection Running...")
    while True:
        text, pred = listen_live_audio()
        time.sleep(0.2)
