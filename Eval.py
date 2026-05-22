import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import joblib

# === Paths ===
dataset_path = r"C:\Users\bogdan\OneDrive\Desktop\Voices"  # Full training dataset
model_path = r"C:\Users\bogdan\OneDrive\Desktop\speaker_model.keras"
encoder_path = r"C:\Users\bogdan\OneDrive\Desktop\label_encoder.joblib"

# === Load model and label encoder ===
model = load_model(model_path)
le = joblib.load(encoder_path)

# === Feature extraction ===
def extract_features(file_path, max_pad_len=100):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]

    return mfcc, y, sr

# === Prediction & Visualization ===
def analyze_prediction(file_path):
    mfcc, y, sr = extract_features(file_path)
    mfcc_input = mfcc[np.newaxis, ..., np.newaxis]

    # Predict
    prediction = model.predict(mfcc_input, verbose=0)[0]
    predicted_index = np.argmax(prediction)
    predicted_label = le.inverse_transform([predicted_index])[0]
    confidence = np.max(prediction) * 100

    # Top 3
    top3_indices = np.argsort(prediction)[-3:][::-1]
    top3_speakers = le.inverse_transform(top3_indices)
    top3_confidences = prediction[top3_indices] * 100

    # === Plot ===
    plt.figure(figsize=(15, 5))

    # Waveform
    plt.subplot(1, 3, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title('Audio Waveform')

    # MFCC
    plt.subplot(1, 3, 2)
    librosa.display.specshow(mfcc, sr=sr, x_axis='time')
    plt.colorbar()
    plt.title('MFCC Features')

    # Confidence scores
    plt.subplot(1, 3, 3)
    plt.barh(range(3), top3_confidences[::-1], color='skyblue')
    plt.yticks(range(3), top3_speakers[::-1])
    plt.xlabel('Confidence (%)')
    plt.title('Top 3 Predictions')
    plt.tight_layout()

    return predicted_label, confidence, top3_speakers, top3_confidences, plt

# === Evaluation from Entire Dataset ===
def interactive_recognition_from_dataset():
    all_files = []
    true_labels = []

    # Load every audio file from all speaker folders
    for speaker in sorted(os.listdir(dataset_path)):
        speaker_path = os.path.join(dataset_path, speaker)
        if not os.path.isdir(speaker_path):
            continue
        for file in os.listdir(speaker_path):
            if file.endswith(".wav"):
                all_files.append(os.path.join(speaker_path, file))
                true_labels.append(speaker)

    print(f"\n🎙️ Loaded {len(all_files)} total audio files from dataset.")

    # Display available files
    for i, (path, label) in enumerate(zip(all_files, true_labels)):
        print(f"{i}: {os.path.basename(path)} ({label})")

    while True:
        try:
            choice = input("\nEnter file number to test (or 'q' to quit): ")
            if choice.lower() == 'q':
                break

            file_num = int(choice)
            if file_num < 0 or file_num >= len(all_files):
                print(f"Please enter a number between 0 and {len(all_files) - 1}")
                continue

            file_path = all_files[file_num]
            true_label = true_labels[file_num]

            speaker, confidence, top3, top3_conf, plot = analyze_prediction(file_path)

            print(f"\n🎧 File: {os.path.basename(file_path)}")
            print(f"📌 True Speaker: {true_label}")
            print(f"🔊 Predicted Speaker: {speaker}")
            print(f"✅ Confidence: {confidence:.2f}%")

            print("\n🏅 Top 3 Predictions:")
            for sp, conf in zip(top3, top3_conf):
                print(f" - {sp}: {conf:.2f}%")

            plot.show()

        except ValueError:
            print("Please enter a valid number or 'q' to quit")

# === Run ===
interactive_recognition_from_dataset()
