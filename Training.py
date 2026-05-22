import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
import collections

# 📁 Dataset path
dataset_path = r"C:\Users\bogdan\OneDrive\Desktop\Voices"

# 🗂️ Filter categories (excluding 'other', '_...')
categories = sorted([
    folder for folder in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, folder)) and not folder.startswith('_') and folder != 'other'
])

print("✅ Speakers found:", categories)

X = []
y = []

# 🔀 Select voice feature: 'mfcc', 'mel', or 'chroma'
feature_type = 'mfcc'  # or 'mel' or 'chroma'

# 🔉 Feature extraction function
def extract_features(file_path, max_pad_len=100):
    y, sr = librosa.load(file_path, sr=None)

    if feature_type == 'mfcc':
        feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    elif feature_type == 'mel':
        feat = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        feat = librosa.power_to_db(feat, ref=np.max)
    elif feature_type == 'chroma':
        stft = np.abs(librosa.stft(y))
        feat = librosa.feature.chroma_stft(S=stft, sr=sr)
    else:
        raise ValueError("Unsupported feature type")

    if feat.shape[1] < max_pad_len:
        pad_width = max_pad_len - feat.shape[1]
        feat = np.pad(feat, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        feat = feat[:, :max_pad_len]

    return feat

# 📥 Load all data
for speaker in categories:
    speaker_folder = os.path.join(dataset_path, speaker)
    for filename in os.listdir(speaker_folder):
        if filename.endswith(".wav"):
            file_path = os.path.join(speaker_folder, filename)
            try:
                features = extract_features(file_path)
                X.append(features)
                y.append(speaker)
            except Exception as e:
                print(f"⚠️ Skipped {file_path}: {e}")

# 🧾 Debug: total samples + count per speaker
print(f"\n🔍 Total samples loaded: {len(X)} (expected: ~{1500 * len(categories)})")
print(f"🧾 Samples per speaker: {collections.Counter(y)}")

# 📊 Prepare dataset
X = np.array(X)
X = X[..., np.newaxis]  # Add channel dimension
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

# 🔀 Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y_cat)

# 🧠 CNN Model
model = Sequential([
    Input(shape=X.shape[1:]),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(categories), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.2, verbose=1)

# 🎯 Evaluation
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n🎯 Model Accuracy on Test Set: {acc * 100:.2f}%")

# 📋 Classification Report
y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(model.predict(X_test), axis=1)
labels = np.arange(len(le.classes_))

print("\n📋 Classification Report:\n")
print(classification_report(y_true, y_pred, labels=labels, target_names=le.classes_, zero_division=0))

# 🔳 Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# 🔍 Predict test folder manually
def predict_audio(test_folder, max_tests=5):
    test_files = sorted([f for f in os.listdir(test_folder) if f.endswith('.wav')])
    print(f"\n🎧 Available test files: {len(test_files)}")

    test_count = 0
    while test_count < max_tests:
        try:
            index = int(input(f"\nSelect an audio index (0 - {len(test_files)-1}) or -1 to exit: "))
            if index == -1:
                break
            if index < 0 or index >= len(test_files):
                print("Invalid index. Try again.")
                continue

            file_path = os.path.join(test_folder, test_files[index])
            mfcc = extract_features(file_path)
            mfcc = mfcc[np.newaxis, ..., np.newaxis]
            prediction = model.predict(mfcc)
            predicted_index = np.argmax(prediction)
            predicted_label = le.inverse_transform([predicted_index])[0]

            print(f"\n📄 File: {test_files[index]}")
            print(f"🔊 Predicted Speaker: {predicted_label}")
            test_count += 1

        except ValueError:
            print("Please enter a valid number.")

    print("\n✅ Testing finished.")

# Example usage:
# predict_audio(r"C:\Users\bogdan\OneDrive\Desktop\test", max_tests=6)
