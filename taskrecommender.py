import tkinter as tk
from tkinter import messagebox
import numpy as np
import joblib
import cv2
import random
from keras.models import model_from_json, Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

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
task_mapping = {
    "positive": [
        "Start writing a short story", "Sketch a picture", "Design a new project idea",
        "Reach out to a friend", "Plan a fun activity", "Join a social group",
        "Go for a run", "Dance to music", "Try a new workout class"
    ],
    "neutral": [
        "Practice deep breathing exercises", "Do a guided meditation session", "Listen to calming nature sounds",
        "Organize your workspace", "Plan your schedule", "Tackle a small project",
        "Read an informative article", "Listen to a podcast", "Take an online course"
    ],
    "negative": [
        "Take a warm bath", "Listen to soothing music", "Spend time in nature",
        "Read a book", "Watch a funny movie", "Do some gentle stretching",
        "Write down your feelings", "Reflect on positive experiences", "Practice gratitude journaling"
    ]
}

emotion_category = {
    "happy": "positive", "joy": "positive", "surprise": "positive",
    "neutral": "neutral",
    "sad": "negative", "anger": "negative", "fear": "negative", "disgust": "negative"
}

def recommend_task(emotion):
    category = emotion_category.get(emotion, "neutral")
    return random.choice(task_mapping[category])

# GUI Setup
root = tk.Tk()
root.title("AI Emotion-Based Task Recommendation System")
root.geometry("500x600")

# Text Emotion Detection
tk.Label(root, text="Enter text for emotion detection:", font=("Arial", 12)).pack(pady=10)
text_entry = tk.Entry(root, width=40)
text_entry.pack(pady=5)

def show_text_emotion():
    raw_text = text_entry.get()
    prediction = pipe_lr.predict([raw_text])[0]
    emoji_icon = emotions_emoji_dict[prediction]
    recommended_task = recommend_task(prediction)
    messagebox.showinfo("Task Recommendation", f"Predicted Emotion: {prediction} {emoji_icon}\nRecommended Task: {recommended_task}")

tk.Button(root, text="Detect Emotion & Recommend Task", command=show_text_emotion).pack(pady=10)

# Face Emotion Detection
def face_emotion_detection():
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        messagebox.showerror("Error", "Could not access webcam.")
        return

    labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
    
    while True:
        i, im = webcam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(im, 1.3, 5)
        
        try:
            for (p, q, r, s) in faces:
                image = gray[q:q+s, p:p+r]
                cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)
                image = cv2.resize(image, (48, 48))
                img = np.array(image).reshape(1, 48, 48, 1) / 255.0
                pred = face_model.predict(img)
                prediction_label = labels[pred.argmax()]
                recommended_task = recommend_task(prediction_label)
                cv2.putText(im, f'{prediction_label}: {recommended_task}', (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))
        
            cv2.imshow("Face Emotion Detection", im)
            if cv2.waitKey(27) & 0xFF == ord('q'):
                break
        except cv2.error:
            pass
    
    webcam.release()
    cv2.destroyAllWindows()

tk.Button(root, text="Start Face Emotion Detection", command=face_emotion_detection).pack(pady=10)

root.mainloop()
