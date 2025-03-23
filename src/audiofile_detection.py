import os
import speech_recognition as sr
from pydub import AudioSegment
from live_call_detection import hybrid_prediction

def convert_audio_to_wav(audio_path):
    audio = AudioSegment.from_file(audio_path)
    wav_path = audio_path.rsplit(".", 1)[0] + ".wav"
    audio.export(wav_path, format="wav")
    return wav_path

def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()

    if not audio_path.lower().endswith(".wav"):
        print("Converting audio to WAV format...")
        audio_path = convert_audio_to_wav(audio_path)

    with sr.AudioFile(audio_path) as source:
        print("Processing audio file...")
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        print("Could not understand the audio.")
        return None
    except sr.RequestError:
        print("Speech Recognition service unavailable.")
        return None

if __name__ == "__main__":
    file_path = input("Enter the path to the audio file: ").strip()

    if not os.path.exists(file_path):
        print("Error: File not found!")
    else:
        text = transcribe_audio(file_path)
        if text:
            print(hybrid_prediction(text))
