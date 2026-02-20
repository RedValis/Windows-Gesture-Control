import time
import math
import cv2
import numpy as np
import mediapipe as mp
import pyautogui
from mediapipe.tasks import python as tasks_python
from mediapipe.tasks.python import vision as mp_vision

# config cam and display ratios :)
MODEL_PATH = "hand_landmarker.task"
CAM_W, CAM_H = 640, 480
CANVAS_W, CANVAS_H = 640, 480
MAX_HANDS = 2

# tuning
SMOOTHING = 0.25
CLICK_CLOSE_THRESH = 0.06
CLICK_DEBOUNCE = 0.25
SWIPE_MIN_DIST = 0.06
VERT_SWIPE_MIN_DIST = 0.08
HOLD_TIME = 0.6
GESTURE_DEBOUNCE = 0.6

def normalized_to_pixel(nx, ny, w, h):
    return int(nx * w), int(ny * h)

def hand_landmarks_to_points(hand_landmarks, w, h):
    pts = []
    for lm in hand_landmarks:
        pts.append((lm.x, lm.y))
    return pts

TIP_IDS = [4, 8, 12, 16, 20]
PIP_IDS = [2, 6, 10, 14, 18]

def fingers_up(hand):
    up = 0
    try:
        thumb_tip = hand[4]
        thumb_ip = hand[2]
        # heuristic for mirrored frame
        if thumb_tip.x < thumb_ip.x:
            up += 1
    except Exception:
        pass
    for t, p in zip(TIP_IDS[1:], PIP_IDS[1:]):
        try:
            if hand[t].y < hand[p].y:
                up += 1
        except Exception:
            pass
    return up

def euclid(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

# mediapipe tasks setup
BaseOptions = tasks_python.BaseOptions
HandLandmarker = mp_vision.HandLandmarker
HandLandmarkerOptions = mp_vision.HandLandmarkerOptions
RunningMode = mp_vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=MAX_HANDS,
    running_mode=RunningMode.VIDEO
)

landmarker = HandLandmarker.create_from_options(options)

# screen
SCREEN_W, SCREEN_H = pyautogui.size()

# camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
if not cap.isOpened():
    raise SystemExit("camera failed")

# state
prev_mouse_x, prev_mouse_y = pyautogui.position()
last_click_time = 0
right_was_closed = False
last_gesture_time = 0
left_last_positions = None
left_hold_start = None

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, ts)

        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

        right_hand = None
        left_hand = None

        if result and result.hand_landmarks:
            hands = list(result.hand_landmarks)
            if len(hands) == 1:
                w_x = hands[0][0].x
                if w_x > 0.5:
                    right_hand = hands[0]
                else:
                    left_hand = hands[0]
            else:
                # assign by wrist x (mirrored feed): larger x -> user's left hand in mirrored frame
                if hands[0][0].x >= hands[1][0].x:
                    right_hand = hands[0]
                    left_hand = hands[1]
                else:
                    right_hand = hands[1]
                    left_hand = hands[0]

        # RIGHT HAND -> mouse + click
        if right_hand:
            idx_tip = right_hand[8]
            sx = max(0.0, min(1.0, idx_tip.x))
            sy = max(0.0, min(1.0, idx_tip.y))
            target_x = int(sx * SCREEN_W)
            target_y = int(sy * SCREEN_H)
            new_x = int(prev_mouse_x + (target_x - prev_mouse_x) * SMOOTHING)
            new_y = int(prev_mouse_y + (target_y - prev_mouse_y) * SMOOTHING)
            try:
                pyautogui.moveTo(new_x, new_y, _pause=False)
            except Exception:
                pyautogui.moveTo(new_x, new_y)
            prev_mouse_x, prev_mouse_y = new_x, new_y

            thumb = (right_hand[4].x, right_hand[4].y)
            index = (right_hand[8].x, right_hand[8].y)
            d = euclid(thumb, index)
            now = time.time()
            closed = d < CLICK_CLOSE_THRESH
            if closed and (not right_was_closed) and (now - last_click_time > CLICK_DEBOUNCE):
                pyautogui.click()
                last_click_time = now
            right_was_closed = closed

            ix, iy = normalized_to_pixel(idx_tip.x, idx_tip.y, CAM_W, CAM_H)
            tx, ty = normalized_to_pixel(thumb[0], thumb[1], CAM_W, CAM_H)
            cv2.circle(frame, (ix, iy), 6, (0,200,0), -1)
            cv2.circle(frame, (tx, ty), 6, (0,200,0), -1)
            cv2.putText(frame, f"R d={d:.3f}", (ix+8, iy-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 1)

        # LEFT HAND -> gestures
        if left_hand:
            n_fingers = fingers_up(left_hand)
            palm_idx = [0,1,5,9,13,17]
            px = float(np.mean([left_hand[i].x for i in palm_idx]))
            pyc = float(np.mean([left_hand[i].y for i in palm_idx]))
            now = time.time()

            if left_last_positions is None:
                left_last_positions = (px, pyc, now, n_fingers)
                left_hold_start = None

            # 4-finger gestures (desktop switching / task view)
            if n_fingers == 4:
                if left_hold_start is None:
                    left_hold_start = now
                    left_last_positions = (px, pyc, now, n_fingers)
                else:
                    dx = px - left_last_positions[0]
                    dy = pyc - left_last_positions[1]
                    abs_dx = abs(dx)
                    abs_dy = abs(dy)
                    # prioritize vertical swipe if vertical movement dominates and exceeds vertical threshold
                    if abs_dy > abs_dx and dy < -VERT_SWIPE_MIN_DIST and (now - last_gesture_time) > GESTURE_DEBOUNCE:
                        # swipe up -> Task View
                        pyautogui.hotkey('winleft', 'tab')
                        last_gesture_time = now
                        left_last_positions = (px, pyc, now, n_fingers)
                        left_hold_start = None
                    # horizontal swipe
                    elif abs_dx > SWIPE_MIN_DIST and (now - last_gesture_time) > GESTURE_DEBOUNCE:
                        # NOTE: mapping inverted per your request:
                        # user swipes left (dx < 0) -> Win+Ctrl+Right
                        # user swipes right (dx > 0) -> Win+Ctrl+Left
                        if dx < 0:
                            pyautogui.hotkey('winleft', 'ctrl', 'right')
                        else:
                            pyautogui.hotkey('winleft', 'ctrl', 'left')
                        last_gesture_time = now
                        left_last_positions = (px, pyc, now, n_fingers)
                        left_hold_start = None
                    else:
                        # small movement + hold -> Task View if held long enough
                        dist = math.hypot(dx, dy)
                        if dist < 0.03 and (now - left_hold_start) > HOLD_TIME and (now - last_gesture_time) > GESTURE_DEBOUNCE:
                            pyautogui.hotkey('winleft', 'tab')
                            last_gesture_time = now
                            left_hold_start = None
                            left_last_positions = (px, pyc, now, n_fingers)

            # 3-finger gestures -> alt+tab
            elif n_fingers == 3:
                if left_last_positions is None:
                    left_last_positions = (px, pyc, now, n_fingers)
                else:
                    dx = px - left_last_positions[0]
                    if abs(dx) > SWIPE_MIN_DIST and (now - last_gesture_time) > GESTURE_DEBOUNCE:
                        pyautogui.keyDown('alt')
                        pyautogui.press('tab')
                        pyautogui.keyUp('alt')
                        last_gesture_time = now
                        left_last_positions = (px, pyc, now, n_fingers)
            else:
                left_last_positions = (px, pyc, now, n_fingers)
                left_hold_start = None

            dbg_x, dbg_y = normalized_to_pixel(px, pyc, CAM_W, CAM_H)
            cv2.putText(frame, f"L f={n_fingers}", (dbg_x-40, dbg_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            cv2.circle(frame, (dbg_x, dbg_y), 6, (255,255,0), -1)

        else:
            left_last_positions = None
            left_hold_start = None

        out = np.hstack((
            cv2.resize(frame, (CAM_W, CAM_H)),
            cv2.resize(canvas, (CANVAS_W, CANVAS_H))
        ))
        cv2.putText(out, "Camera", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(out, "mouse and gestures", (CAM_W+20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.imshow("hand mouse gestures", out)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    landmarker.__del__()
