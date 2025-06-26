# GESTURE-BASED-CURSOR
This system transforms hand gestures into real-time cursor movements and actions (clicks, scrolling, dragging) using computer vision and machine learning. It's designed for touchless interaction with computers, ideal for presentations, gaming, or accessibility applications.

Core Features
1. Cursor Movement
How it works:

Tracks your hand's 3D position (via MediaPipe's 21 hand landmarks).

Maps hand coordinates to screen coordinates with adjustable speed/smoothing.

Gesture: Open palm facing the camera.

Technical: Uses pyautogui.moveTo() with exponential smoothing for fluid motion.

2. Click Actions
Gesture	Action	Technical Implementation
✊ Closed fist	Left Click	pyautogui.click()
✌ Peace sign	Right Click	pyautogui.rightClick()
👍 Thumbs up	Scroll Up	pyautogui.scroll(40)
👎 Thumbs down	Scroll Down	pyautogui.scroll(-40)
🤏 Pinch	Drag & Drop	pyautogui.mouseDown()/mouseUp()

you can checkout the cursor speed and smoothness as per your use

Limitations & Solutions
Limitation	Solution
Requires training	Pre-train with diverse hand sizes/angles
Lighting sensitivity	Use IR camera or controlled lighting
Occlusions	Multi-camera setup for robustness
