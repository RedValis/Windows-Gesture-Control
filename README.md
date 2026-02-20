# Windows Gesture Control

Control your Windows PC using hand gestures via your webcam.

This project uses MediaPipe Tasks (Hand Landmarker) and OpenCV to track hands in real-time and map gestures to mouse control and Windows navigation commands.

---

## Features

### Mouse Control (Right Hand)

- Index finger tip controls mouse movement
- Thumb and index finger pinch performs a left click
- Built-in smoothing for stable cursor movement

**Implementation:**  
`gesture_control.py`

---

### Windows Navigation Gestures (Left Hand)

#### Four Fingers Up

- Swipe up → Task View (`Win + Tab`)
- Swipe left → Next virtual desktop (`Win + Ctrl + Right`)
- Swipe right → Previous virtual desktop (`Win + Ctrl + Left`)
- Hold steady → Task View

#### Three Fingers Up

- Horizontal swipe → `Alt + Tab`

Gesture detection is based on:

- Finger counting
- Palm center tracking
- Movement thresholds
- Gesture debounce timing

**Implementation:**  
`gesture_control.py`

---

### Hand Mirror / Debug Mode

A secondary script renders a stylized mirrored hand using MediaPipe landmarks.

**File:**  
`hand_landmarker_model.py`

Useful for:

- Debugging landmark detection
- Demonstrations
- Visualizing hand tracking behavior

---

## Requirements

From `requirements.txt`:
mediapipe
opencv-python
pyautogui
numpy


Additional requirements:

- Python 3.9–3.11 recommended
- Windows OS (uses Windows-specific hotkeys)
- Webcam

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/Windows-Gesture-Control.git
cd Windows-Gesture-Control
```

### 2. Install dependencies
```pip install -r requirements.txt```

### 3. Ensure the MediaPipe model file is present

The following file must exist in the project root:
hand_landmarker.task

## Running the Application
### Main Gesture Controller
```python gesture_control.py```
Press ESC to exit.

### Mirror / Debug Mode
```python hand_landmarker_model.py```
Press ESC to exit.
