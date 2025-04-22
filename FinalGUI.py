import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import QDateTime
from pymongo import MongoClient
import joblib
import cv2
import numpy as np
import random
import sys
import matplotlib.pyplot as plt
from keras.models import model_from_json, Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["MoodTracker"]
mood_collection = db["mood_logs"]

# Load Text Emotion Detection Model
pipe_lr = joblib.load("C:/Users/likes/OneDrive/portfolio/Desktop/Text Emotion Detection/text_emotion.pkl")
emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"}

# Load Face Emotion Model
json_file = open("C:/Users/likes/OneDrive/portfolio/Desktop/Face Emotion Detection/facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
face_model = model_from_json(model_json, custom_objects={'Sequential': Sequential, 'Conv2D': Conv2D, 'MaxPooling2D': MaxPooling2D, 'Dropout': Dropout, 'Flatten': Flatten, 'Dense': Dense})
face_model.load_weights("C:/Users/likes/OneDrive/portfolio/Desktop/Face Emotion Detection/facialemotionmodel.h5")

# Initialize face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Task Recommendation Categories
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

# Email sending function for stress alerts
def send_email_alert(employee_id, recipient_email):
    sender_email = "likesraghu979@gmail.com"  # Replace with your email
    app_password = "wwdq ckfz snwu wmnz"  # Replace with your App Password

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = "🚨 Employee Stress Alert"

    body = f"⚠️ Alert: Employee {employee_id} is experiencing prolonged stress. Immediate action is recommended."
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, app_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()
        print(f"✅ Email successfully sent to {recipient_email}!")
    except Exception as e:
        print(f"❌ Failed to send email: {str(e)}")

# Function to check for prolonged stress
def check_for_prolonged_stress(employee_id, mood_data, threshold_days=3, mood_threshold=2):
    today = datetime.today()
    stress_days = 0

    # Check mood data for the last 7 days
    for record in mood_data:
        mood_date = datetime.strptime(record["date"], "%Y-%m-%d")
        if today - mood_date <= timedelta(days=7):
            if record["mood"] <= mood_threshold:
                stress_days += 1

    # If stress exceeds the threshold, trigger email alert
    if stress_days >= threshold_days:
        print(f"Prolonged stress detected for employee {employee_id}. Sending email alert to HR...")
        send_email_alert(employee_id, "likesraghs979@gmail.com")  # Replace with HR's email

# Function to recommend task based on emotion
def recommend_task(emotion):
    category = emotion_category.get(emotion, "neutral")
    return random.choice(task_mapping[category])

class EmotionApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("AI Emotion Tracking & Task Recommendation")
        self.setGeometry(100, 100, 600, 400)
        
        layout = QtWidgets.QVBoxLayout()
        
        # Text Emotion Detection
        self.text_label = QtWidgets.QLabel("Enter text for emotion detection:")
        layout.addWidget(self.text_label)
        
        self.text_entry = QtWidgets.QLineEdit()
        layout.addWidget(self.text_entry)
        
        self.text_button = QtWidgets.QPushButton("Detect Emotion & Recommend Task")
        self.text_button.clicked.connect(self.show_text_emotion)
        layout.addWidget(self.text_button)
        
        # Face Emotion Detection
        self.face_button = QtWidgets.QPushButton("Start Face Emotion Detection")
        self.face_button.clicked.connect(self.face_emotion_detection)
        layout.addWidget(self.face_button)
        
        # Mood Tracking
        self.mood_label = QtWidgets.QLabel("Select Your Mood:")
        layout.addWidget(self.mood_label)
        
        self.moodComboBox = QtWidgets.QComboBox()
        self.moodComboBox.addItems(["Happy", "Neutral", "Sad", "Stressed", "Angry", "Surprise", "Disgust"])
        layout.addWidget(self.moodComboBox)
        
        self.submitButton = QtWidgets.QPushButton("Log Mood")
        self.submitButton.clicked.connect(self.log_mood)
        layout.addWidget(self.submitButton)
        
        self.viewTrendsButton = QtWidgets.QPushButton("View Mood Trends")
        self.viewTrendsButton.clicked.connect(self.show_mood_trends)
        layout.addWidget(self.viewTrendsButton)

        self.check_stress_button = QtWidgets.QPushButton("Check for Prolonged Stress")
        self.check_stress_button.clicked.connect(self.check_stress)
        layout.addWidget(self.check_stress_button)
        
        self.setLayout(layout)

    def show_text_emotion(self):
        raw_text = self.text_entry.text()
        prediction = pipe_lr.predict([raw_text])[0]
        emoji_icon = emotions_emoji_dict[prediction]
        recommended_task = recommend_task(prediction)
        QtWidgets.QMessageBox.information(self, "Task Recommendation", f"Predicted Emotion: {prediction} {emoji_icon}\nRecommended Task: {recommended_task}")
    
    def face_emotion_detection(self):
        webcam = cv2.VideoCapture(0)
        if not webcam.isOpened():
            QtWidgets.QMessageBox.warning(self, "Error", "Could not access webcam.")
            return
        
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
                cv2.putText(frame, f'{label}: {task}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            cv2.imshow("Face Emotion Detection", frame)
            if cv2.waitKey(27) & 0xFF == ord('q'):
                break
        
        webcam.release()
        cv2.destroyAllWindows()

    def log_mood(self):
        mood = self.moodComboBox.currentText()
        mood_collection.insert_one({"mood": mood, "timestamp": QDateTime.currentDateTime().toString()})
        QtWidgets.QMessageBox.information(self, "Success", "Mood logged successfully!")
    
    def show_mood_trends(self):
        moods = {self.moodComboBox.itemText(i): 0 for i in range(self.moodComboBox.count())}

        for record in mood_collection.find():
            if record["mood"] in moods:
                moods[record["mood"]] += 1
        plt.bar(moods.keys(), moods.values())
        plt.show()

    def check_stress(self):
        employee_id = "E12345"  # You can modify this to select dynamically
        today = datetime.today()
        stress_days = 0
        mood_threshold = 2
        threshold_days = 3
        # Retrieve mood data from MongoDB 
        mood_data = list(mood_collection.find({"employee_id": employee_id}))
        for record in mood_data:
            mood_date = datetime.strptime(record["timestamp"], "%Y-%m-%d %H:%M:%S")
            if today - mood_date <= timedelta(days=7):
                if record["mood"] in ["Stressed", "Sad", "Angry"]:
                    stress_days += 1
        
        if stress_days >= threshold_days:
            send_email_alert(employee_id, "likesraghs979@gmail.com")  # Replace with HR email
            QtWidgets.QMessageBox.warning(self, "Prolonged Stress", f"Prolonged stress detected for {employee_id}. HR has been notified!")
        else:
            QtWidgets.QMessageBox.information(self, "No Stress Alert", f"{employee_id} is not showing prolonged stress.")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = EmotionApp()
    window.show()
    sys.exit(app.exec_())
