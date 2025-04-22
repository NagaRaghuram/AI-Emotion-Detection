import sys
import librosa
import numpy as np
import sounddevice as sd
import tensorflow as tf
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel

# Load trained model
try:
    model = tf.keras.models.load_model("speech_emotion_recognition_model.h5")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ ERROR: Could not load model - {e}")
    sys.exit(1)

# Define emotions list (same order as training)
EMOTIONS = ['Anger', 'Boredom', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Neutral']

# Function to extract MFCC features from audio
def extract_mfcc_from_audio(audio, sr, max_length=128):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

    print(f"🔍 DEBUG: Raw MFCC shape: {mfcc.shape}")  # Debug output

    # Normalize MFCC
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)

    # Padding or truncation
    if mfcc.shape[1] < max_length:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_length - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_length]

    print(f"✅ Processed MFCC shape: {mfcc.shape}")  # Debug output

    return mfcc

# GUI Class
class SpeechEmotionGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Speech Emotion Recognition")
        self.setGeometry(100, 100, 400, 200)

        # Layout
        layout = QVBoxLayout()

        # Push-to-Talk Button
        self.record_button = QPushButton("🎤 Push to Talk", self)
        self.record_button.setStyleSheet("font-size: 18px; padding: 10px;")
        self.record_button.pressed.connect(self.start_recording)

        # Label to Display Emotion
        self.result_label = QLabel("🎤 Speak to detect emotion...", self)
        self.result_label.setStyleSheet("font-size: 16px; color: blue;")

        # Add widgets to layout
        layout.addWidget(self.record_button)
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def start_recording(self):
        duration = 3  # Record for 3 seconds
        sr = 22050  # Sample rate

        self.result_label.setText("⏳ Listening... Speak Now!")
        QApplication.processEvents()  # Update UI immediately

        # Record audio
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        audio = np.squeeze(audio)  # Convert from 2D to 1D array

        print(f"🔊 Recorded Audio Length: {len(audio)} samples")  # Debug output

        # Extract MFCC features
        mfcc = extract_mfcc_from_audio(audio, sr)
        mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dimension
        mfcc = np.expand_dims(mfcc, axis=-1)  # Add channel dimension (needed for CNN)

        print(f"🚀 Final MFCC Shape Sent to Model: {mfcc.shape}")  # Debug output

        try:
            # Predict emotion
            prediction = model.predict(mfcc)
            
            print(f"✅ Model Prediction Raw Output: {prediction}")  # Debug output

            predicted_label = EMOTIONS[np.argmax(prediction)]
            
            print(f"🎯 Predicted Emotion: {predicted_label}")  # Debug output

            # Update label in UI
            self.result_label.setText(f"🗣️ Detected Emotion: {predicted_label}")

        except Exception as e:
            print(f"❌ ERROR: {e}")
            self.result_label.setText("⚠️ Error in prediction")

# Run GUI
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = SpeechEmotionGUI()
    gui.show()
    sys.exit(app.exec_())
