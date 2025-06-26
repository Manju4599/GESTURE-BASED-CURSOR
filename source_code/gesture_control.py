import cv2
import joblib
import numpy as np
import mediapipe as mp
import pyautogui
import time
from sklearn.ensemble import RandomForestClassifier

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# Cursor control parameters
cursor_smoothing = 0.5
prev_x, prev_y = 0, 0
cursor_speed = 1.0

# Gesture mapping
GESTURES = {
    0: "move_cursor",
    1: "left_click",
    2: "right_click",
    3: "scroll_up",
    4: "scroll_down",
    5: "drag"
}

# Initialize webcam
cap = cv2.VideoCapture(0)

# Data collection and model training functions
def collect_gesture_data(gesture_class, num_samples=30):
    """Collect training data for gestures"""
    data = []
    print(f"Collecting {num_samples} samples for {GESTURES[gesture_class]}...")
    
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7
    ) as hands:
        
        collected = 0
        while collected < num_samples:
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                landmarks = []
                for landmark in results.multi_hand_landmarks[0].landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                cv2.putText(frame, f"Collected: {collected}/{num_samples}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Data Collection', frame)
                
                key = cv2.waitKey(1)
                if key == ord('c'):
                    data.append(landmarks)
                    collected += 1
                elif key == ord('q'):
                    break
                    
    return np.array(data), np.full(num_samples, gesture_class)

def train_gesture_model():
    """Train the gesture recognition model"""
    try:
        # Try to load existing model
        model = joblib.load('gesture_model.pkl')
        print("Loaded pre-trained model")
        return model
    except:
        print("Training new model...")
        # Collect training data
        X, y = [], []
        for gesture_id in GESTURES.keys():
            data, labels = collect_gesture_data(gesture_id, 30)
            X.append(data)
            y.append(labels)
        
        X = np.vstack(X)
        y = np.hstack(y)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X, y)
        joblib.dump(model, 'gesture_model.pkl')
        return model

# Initialize model
model = train_gesture_model()

# Main interaction loop
def main_interaction_loop():
    global prev_x, prev_y
    
    drag_mode = False
    last_click_time = 0
    click_cooldown = 0.5  # seconds
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        current_time = time.time()
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get landmarks for prediction
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            # Predict gesture
            gesture_id = model.predict([landmarks])[0]
            gesture_name = GESTURES[gesture_id]
            
            # Get hand center (for cursor control)
            cx, cy = int(hand_landmarks.landmark[9].x * frame.shape[1]), int(hand_landmarks.landmark[9].y * frame.shape[0])
            
            # Map hand position to screen coordinates
            screen_x = np.interp(hand_landmarks.landmark[9].x, [0, 1], [0, screen_width])
            screen_y = np.interp(hand_landmarks.landmark[9].y, [0, 1], [0, screen_height])
            
            # Smooth cursor movement
            smooth_x = prev_x + (screen_x - prev_x) * cursor_smoothing
            smooth_y = prev_y + (screen_y - prev_y) * cursor_smoothing
            
            # Perform actions based on gesture
            if gesture_name == "move_cursor":
                pyautogui.moveTo(smooth_x * cursor_speed, smooth_y * cursor_speed)
                prev_x, prev_y = smooth_x, smooth_y
                
            elif gesture_name == "left_click" and (current_time - last_click_time) > click_cooldown:
                pyautogui.click()
                last_click_time = current_time
                
            elif gesture_name == "right_click" and (current_time - last_click_time) > click_cooldown:
                pyautogui.rightClick()
                last_click_time = current_time
                
            elif gesture_name == "scroll_up":
                pyautogui.scroll(40)
                
            elif gesture_name == "scroll_down":
                pyautogui.scroll(-40)
                
            elif gesture_name == "drag":
                if not drag_mode:
                    pyautogui.mouseDown()
                    drag_mode = True
                pyautogui.moveTo(smooth_x * cursor_speed, smooth_y * cursor_speed)
                prev_x, prev_y = smooth_x, smooth_y
            else:
                if drag_mode:
                    pyautogui.mouseUp()
                    drag_mode = False
            
            # Display gesture info
            cv2.putText(frame, f"Gesture: {gesture_name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        else:
            if drag_mode:
                pyautogui.mouseUp()
                drag_mode = False
        
        cv2.imshow('Gesture Control', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_interaction_loop()
