# Fraudulent Call Detection

## 📌 Project Overview
This project detects fraudulent calls in real time using **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques. It processes live audio or pre-recorded files to classify as **Fraud** or **Normal**, analysing speech context.

## 🚀 Features
- **Real-Time Call Monitoring**: Analyses ongoing calls at regular intervals.
- **Audio File Analysis**: Supports pre-recorded audio input for fraud detection.
- **NLP Processing**: Uses TF-IDF and Word2Vec for text feature rxtraction.
- **Machine Learning Classification**: Implements **Random Forest** for high accuracy.

## ⚙️ Setup & Installation
### Prerequisites
- **Python 3.9+**
- **Pip**
- **ffmpeg** (only for audiofile detection, if not required delete the audiofile_detection.py file and it should run smoothly)

### Steps to Run

#### 1. Install dependencies
```sh
pip install -r requirements.txt
```

#### 2. Install globally by running following code in a python file
```sh
nltk.download('stopwords')
nltk.download('punkt')
```

#### 3. Go to Source directory
```sh
cd src
```

#### 4. Train the model
```sh
python train_model.py
```

#### 5. Run real-time fraud detection
```sh
python live_call_detection.py
```

#### 6. Run fraud detection on an audio file
```sh
python audiofile_detection.py
```

## 📊 Model Training Pipeline
- **Data Preprocessing**: Cleans transcripts and extracts text features.
- **Feature Extraction**: Converts text into numerical vectors using **TF-IDF** and **Word2Vec**.
- **Model Training**: Trains a **Random Forest Classifier** for classification.
- **Evaluation**: Computes accuracy, precision, recall, and F1-score.

## 🎤 Live Call Detection
- **Input**: Continuous live audio stream.
- **Processing**: Converts speech to text using **SpeechRecognition**.
- **Prediction**: Classifies every **n-seconds** segment (Default n = 10).
- **Output**: Displays if fraud in real-time.

## 🛠️ Technologies Used
- **Python** (SpeechRecognition, Scikit-learn, NLTK)
- **NLP** (TF-IDF, Word2Vec, Stopword Removal)
- **Machine Learning** (Random Forest Classifier)

## 📌 Future Enhancements
- Integration into existing **Telecom System**.
- Add **Deep Learning Models** (LSTMs, Transformers) for improved accuracy.
- Integrate **Speaker Verification** for fraud analysis.

## 📜 License
This project is licensed under the **MIT License**.
