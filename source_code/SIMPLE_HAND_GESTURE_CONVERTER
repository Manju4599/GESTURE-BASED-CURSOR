import cv2
import numpy as np
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os
import csv

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Gesture mapping
gesture_map = {
    0: "Zero",
    1: "One",
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five"
}

# Data collection functions
def collect_real_time_data(gesture_class, num_samples=30):
    data = []
    print(f"Collecting {num_samples} samples for '{gesture_map[gesture_class]}'. Show gesture and press 'c' to capture.")
    
    collected = 0
    while collected < num_samples:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                cv2.putText(frame, f"Collected: {collected}/{num_samples}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Data Collection', frame)
                
                key = cv2.waitKey(1)
                if key == ord('c'):
                    data.append(landmarks)
                    collected += 1
                    print(f"Saved sample {collected}/{num_samples}")
                elif key == ord('q'):
                    return None
    
    return np.array(data), np.full(num_samples, gesture_class)

def save_to_csv(data, labels, filename="hand_gestures.csv"):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        for row, label in zip(data, labels):
            writer.writerow([label] + list(row))
    print(f"Data saved to {filename}")

def load_from_csv(filename="hand_gestures.csv"):
    if not os.path.exists(filename):
        return None, None
    
    data = []
    labels = []
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            labels.append(int(row[0]))
            data.append([float(x) for x in row[1:]])
    
    return np.array(data), np.array(labels)

# Training function
def train_model():
    X, y = load_from_csv()
    
    if X is None or len(np.unique(y)) < 3:  # Require at least 3 classes
        print("No/incomplete training data found. Let's collect some!")
        X, y = np.array([]).reshape(0, 63), np.array([])  # 21 landmarks * 3 values
        
        for gesture in gesture_map.keys():
            data, labels = collect_real_time_data(gesture, num_samples=30)
            if data is None:
                return None
            X = np.vstack([X, data]) if X.size else data
            y = np.concatenate([y, labels]) if y.size else labels
            save_to_csv(data, labels)
    
    if X.size == 0:
        return None
    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    print("Model trained successfully!")
    return model

# Main function
def main():
    model = train_model()
    if model is None:
        print("Failed to train model. Exiting.")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                proba = model.predict_proba([landmarks])[0]
                max_proba = np.max(proba)
                
                if max_proba > 0.7:
                    detected_gesture = model.predict([landmarks])[0]
                    text = f"{gesture_map[detected_gesture]} ({max_proba:.0%})"
                    color = (0, 255, 0)
                else:
                    text = "?"
                    color = (0, 0, 255)
                
                cv2.putText(frame, text, (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow('Sign Language Converter', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
