import os
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, TimeDistributed, Flatten, Dense, Dropout, Reshape
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# 📌 EMOTION MAPPING
EMOTIONS = {
    "W": "Angry",
    "L": "Boredom",
    "E": "Disgust",
    "A": "Fear",
    "F": "Happy",
    "T": "Sad",
    "N": "Neutral"
}

# 📌 Load Dataset Path
DATASET_PATH = r"C:/Users/likes/OneDrive/portfolio/Desktop/Speech Emotion Detection/emo_db"  # Change to your dataset path

# 📌 Extract Features (MFCC)
def extract_features(file_path, max_length=128):
    audio, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

    # Pad or truncate to max_length
    if mfcc.shape[1] < max_length:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_length - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_length]

    return mfcc

# 📌 Load Dataset
X_features, y_labels = [], []

for file in os.listdir(DATASET_PATH):
    if file.endswith(".wav"):
        emotion_label = file[5]  # Extract emotion from filename (Berlin dataset format)
        if emotion_label in EMOTIONS:
            file_path = os.path.join(DATASET_PATH, file)
            features = extract_features(file_path)
            X_features.append(features)
            y_labels.append(list(EMOTIONS.keys()).index(emotion_label))  # Convert to index

# 📌 Convert to numpy arrays
X_features = np.array(X_features)
y_labels = np.array(y_labels)

# 📌 Normalize Features
X_features = X_features / np.max(np.abs(X_features))

# 📌 Reshape for CNN (Add Channel Dimension)
X_features = np.expand_dims(X_features, axis=-1)

# 📌 One-Hot Encode Labels
y_labels = to_categorical(y_labels, num_classes=len(EMOTIONS))

# 📌 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=42)

print(f"✅ Training Samples: {X_train.shape}, Testing Samples: {X_test.shape}")

# 📌 Define CNN + LSTM Model
model = Sequential([
    Reshape((40, 128, 1), input_shape=(40, 128, 1)),  # Reshape for CNN

    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),

    TimeDistributed(Flatten()),  # Convert CNN output to time-distributed input for LSTM
    LSTM(64, return_sequences=False),

    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(EMOTIONS), activation='softmax')
])

# 📌 Compile Model
model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

# 📌 Train Model
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))

# 📌 Save Model
model.save("speech_emotion_recognition_model.h5")
print("✅ Model Training Completed & Saved as 'speech_emotion_recognition_model.h5'!")
