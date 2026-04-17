# 🖐️ Gesture-Based Cursor Control

Transform your hand gestures into real-time cursor movements and actions using cutting-edge **computer vision** and **machine learning**! This system enables touchless interaction with computers, making it perfect for presentations, gaming, accessibility applications, and remote control scenarios.

## 🎯 Overview

This innovative gesture recognition system uses **MediaPipe** for hand detection and **scikit-learn** for machine learning to accurately identify and respond to hand gestures. Control your computer without touching the mouse or keyboard—just use natural hand movements!

### Key Highlights

- ✋ **Real-time Hand Gesture Recognition** using MediaPipe's 21-point hand landmark detection
- 🖱️ **Fluid Cursor Control** with exponential smoothing for natural movement
- 🎯 **Multiple Gesture Actions** including clicks, scrolling, and dragging
- 🤖 **ML-Based Classification** with Random Forest for reliable gesture identification
- 🎮 **Accessibility Friendly** - perfect for users who need touchless interaction
- 📹 **Webcam-Based Input** - works with any standard webcam

## 📋 Table of Contents

- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Supported Gestures](#supported-gestures)
- [Configuration & Customization](#configuration--customization)
- [Limitations & Solutions](#limitations--solutions)
- [Future Improvements](#future-improvements)
- [Troubleshooting](#troubleshooting)

## ✨ Features

### 1. Cursor Movement
- **Gesture:** Open palm facing the camera
- **Action:** Smooth, real-time cursor tracking
- **Technical Implementation:** 
  - Tracks hand's 3D position via MediaPipe's 21 hand landmarks
  - Maps hand coordinates to screen coordinates using `numpy.interp()`
  - Applies exponential smoothing for fluid motion
  - Adjustable cursor speed and smoothing parameters

### 2. Click Actions
| Gesture | Action | Technical Implementation |
|---------|--------|--------------------------|
| ✊ Closed Fist | Left Click | `pyautogui.click()` |
| ✌️ Peace Sign | Right Click | `pyautogui.rightClick()` |
| 👍 Thumbs Up | Scroll Up | `pyautogui.scroll(40)` |
| 👎 Thumbs Down | Scroll Down | `pyautogui.scroll(-40)` |
| 🤏 Pinch Gesture | Drag & Drop | `pyautogui.mouseDown()/mouseUp()` |

### 3. Machine Learning Model
- **Algorithm:** Random Forest Classifier (100 estimators)
- **Features:** 63-dimensional hand landmark vectors (21 landmarks × 3 coordinates)
- **Training:** Collects real-time gesture samples with user validation
- **Persistence:** Model saved as `gesture_model.pkl` for future use

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Language** | Python 3.7+ | Core implementation |
| **Vision** | OpenCV | Webcam frame capture & display |
| **Hand Detection** | MediaPipe | 21-point hand landmark detection |
| **ML Framework** | scikit-learn | Random Forest classification |
| **Automation** | PyAutoGUI | Cursor and keyboard control |
| **Data Processing** | NumPy | Numerical calculations |
| **Data Storage** | Pandas/CSV | Gesture data storage |

## 📁 Project Structure
