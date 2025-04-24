import sys
import smtplib
import librosa
import sounddevice as sd
import numpy as np
import random
import tensorflow as tf
import joblib
import cv2
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pymongo import MongoClient
from PyQt5 import QtWidgets
from PyQt5.QtCore import QDateTime
from keras.models import model_from_json, Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# -------------------- MODEL & DB LOADING -------------------- #

# MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["MoodTracker"]
mood_collection = db["mood_logs"]

# Load Text Emotion Model
pipe_lr = joblib.load("C:/Users/likes/OneDrive/portfolio/Desktop/AI Emotion Detection/Text Emotion Detection/text_emotion.pkl")
emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"}

# Load Face Emotion Model
# ✅ Correctly loading Keras model (HDF5 binary format)
face_model = tf.keras.models.load_model("C:/Users/likes/OneDrive/portfolio/Desktop/AI Emotion Detection/Face Emotion Detection/facialemotionmodel.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load Speech Emotion Recognition Model
speech_model = tf.keras.models.load_model("C:/Users/likes/OneDrive/portfolio/Desktop/AI Emotion Detection/Speech Emotion Detection/speech_emotion_recognition_model.h5")
speech_emotions = ['Anger', 'Boredom', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Neutral']

# -------------------- HELPERS -------------------- #

def extract_mfcc_from_audio(audio, sr, max_length=128):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
    mfcc = np.pad(mfcc, ((0, 0), (0, max(0, max_length - mfcc.shape[1]))), mode='constant')[:, :max_length]
    return mfcc

emotion_category = {
    "happy": "positive", "joy": "positive", "surprise": "positive",
    "neutral": "neutral",
    "sad": "negative", "anger": "negative", "fear": "negative", "disgust": "negative"
}

task_mapping = {
    "positive": ["Start writing a story", "Sketch a picture", "Join a social group"],
    "neutral": ["Meditation", "Plan your schedule", "Read an article"],
    "negative": ["Take a warm bath", "Listen to music", "Write your feelings"]
}

def recommend_task(emotion):
    category = emotion_category.get(emotion.lower(), "neutral")
    return random.choice(task_mapping[category])

def send_email_alert(employee_id, recipient_email):
    sender_email = "likesraghu979@gmail.com"
    app_password = "wwdq ckfz snwu wmnz"

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = "🚨 Employee Stress Alert"
    msg.attach(MIMEText(f"⚠️ Alert: Employee {employee_id} is experiencing prolonged stress. Immediate action is recommended.", "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, app_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()
        print(f"✅ Email sent to {recipient_email}!")
    except Exception as e:
        print(f"❌ Email failed: {e}")

# -------------------- VIRTUAL ASSISTANT -------------------- #

def virtual_assistant(query):
    # Predefined responses based on user queries
    responses = {
        "hello": "Hi there! How can I assist you today?",
        "how are you": "I'm just a bot, but I'm here to help you!",
        "what is your name": "I am your virtual assistant, here to make your day better!",
        "I need help":"You can ask me about your mood, tasks, or get recommendations. Just type your query.",
        "help": "You can ask me about your mood, tasks, or get recommendations. Just type your query.",
        "bye": "Goodbye! Have a great day!"
    }
    
    # Return a response based on the query or a default message
    return responses.get(query.lower(), "Sorry, I can't understand that.")

# -------------------- MAIN GUI -------------------- #

class EmotionApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("AI Emotion Tracking & Task Optimizer")
        self.setGeometry(100, 100, 600, 500)

        layout = QtWidgets.QVBoxLayout()

        # Text Emotion
        self.text_label = QtWidgets.QLabel("Enter text:")
        layout.addWidget(self.text_label)

        self.text_entry = QtWidgets.QLineEdit()
        layout.addWidget(self.text_entry)

        self.text_button = QtWidgets.QPushButton("Analyze Text Emotion")
        self.text_button.clicked.connect(self.show_text_emotion)
        layout.addWidget(self.text_button)

        # Face Emotion
        self.face_button = QtWidgets.QPushButton("Start Face Emotion Detection")
        self.face_button.clicked.connect(self.face_emotion_detection)
        layout.addWidget(self.face_button)

        # Speech Emotion
        self.speech_button = QtWidgets.QPushButton("🎤 Push to Talk (Speech Emotion)")
        self.speech_button.clicked.connect(self.speech_emotion_recognition)
        layout.addWidget(self.speech_button)

        # Mood Tracker
        self.moodComboBox = QtWidgets.QComboBox()
        self.moodComboBox.addItems(["Happy", "Neutral", "Sad", "Stressed", "Angry", "Surprise", "Disgust"])
        layout.addWidget(QtWidgets.QLabel("Log Your Mood:"))
        layout.addWidget(self.moodComboBox)

        self.submitButton = QtWidgets.QPushButton("Submit Mood")
        self.submitButton.clicked.connect(self.log_mood)
        layout.addWidget(self.submitButton)

        self.viewTrendsButton = QtWidgets.QPushButton("View Mood Trends")
        self.viewTrendsButton.clicked.connect(self.show_mood_trends)
        layout.addWidget(self.viewTrendsButton)

        self.check_stress_button = QtWidgets.QPushButton("Check Prolonged Stress")
        self.check_stress_button.clicked.connect(self.check_stress)
        layout.addWidget(self.check_stress_button)

        # Virtual Assistant
        self.chat_label = QtWidgets.QLabel("Ask the Virtual Assistant:")
        layout.addWidget(self.chat_label)

        self.chat_input = QtWidgets.QLineEdit()
        layout.addWidget(self.chat_input)

        self.chat_button = QtWidgets.QPushButton("Ask Assistant")
        self.chat_button.clicked.connect(self.ask_virtual_assistant)
        layout.addWidget(self.chat_button)

        self.setLayout(layout)

    def show_text_emotion(self):
        raw_text = self.text_entry.text()
        prediction = pipe_lr.predict([raw_text])[0]
        emoji_icon = emotions_emoji_dict[prediction]
        recommended_task = recommend_task(prediction)
        QtWidgets.QMessageBox.information(self, "Task Recommendation", f"Predicted Emotion: {prediction} {emoji_icon}\nRecommended Task: {recommended_task}")

    def face_emotion_detection(self):
        webcam = cv2.VideoCapture(0)
        labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
        while True:
            _, frame = webcam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                roi = gray[y:y+h, x:x+w]
                roi = cv2.resize(roi, (48, 48))
                img = np.array(roi).reshape(1, 48, 48, 1) / 255.0
                pred = face_model.predict(img)
                label = labels[pred.argmax()]
                task = recommend_task(label)
                cv2.putText(frame, f'{label}: {task}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.imshow("Face Emotion Detection", frame)
            if cv2.waitKey(27) & 0xFF == ord('q'):
                break
        webcam.release()
        cv2.destroyAllWindows()

    def speech_emotion_recognition(self):
        duration = 3
        sr = 22050
        QtWidgets.QMessageBox.information(self, "Recording", "Listening... Speak now!")
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        audio = np.squeeze(audio)
        mfcc = extract_mfcc_from_audio(audio, sr)
        mfcc = np.expand_dims(np.expand_dims(mfcc, axis=0), axis=-1)
        try:
            prediction = speech_model.predict(mfcc)
            predicted_label = speech_emotions[np.argmax(prediction)]
            recommended_task = recommend_task(predicted_label)
            QtWidgets.QMessageBox.information(self, "Speech Emotion", f"Detected: {predicted_label}\nSuggested Task: {recommended_task}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Speech emotion prediction failed: {e}")

    def log_mood(self):
        mood = self.moodComboBox.currentText()
        mood_collection.insert_one({"employee_id": "E12345", "mood": mood, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        QtWidgets.QMessageBox.information(self, "Success", "Mood logged!")

    def show_mood_trends(self):
        mood_counts = {}
        for record in mood_collection.find():
            mood = record.get("mood")
            mood_counts[mood] = mood_counts.get(mood, 0) + 1
        plt.bar(mood_counts.keys(), mood_counts.values())
        plt.xlabel("Mood")
        plt.ylabel("Frequency")
        plt.title("Mood Trends")
        plt.show()

    def check_stress(self):
        employee_id = "E12345"
        today = datetime.today()
        stress_days = 0
        mood_data = list(mood_collection.find({"employee_id": employee_id}))
        for record in mood_data:
            mood_date = datetime.strptime(record["timestamp"], "%Y-%m-%d %H:%M:%S")
            if today - mood_date <= timedelta(days=7):
                if record["mood"] in ["Stressed", "Sad", "Angry"]:
                    stress_days += 1
        if stress_days >= 3:
            send_email_alert(employee_id, "HR@gmail")
            QtWidgets.QMessageBox.warning(self, "Alert", f"Prolonged stress detected for {employee_id}. HR notified.")
        else:
            QtWidgets.QMessageBox.information(self, "OK", f"No prolonged stress detected for {employee_id}.")

    def ask_virtual_assistant(self):
        query = self.chat_input.text()
        response = virtual_assistant(query)
        QtWidgets.QMessageBox.information(self, "Assistant Response", response)

# -------------------- APP RUN -------------------- #

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = EmotionApp()
    window.show()
    sys.exit(app.exec_())

