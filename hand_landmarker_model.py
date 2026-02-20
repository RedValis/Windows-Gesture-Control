"""
mirror_hand_tasks.py
hand mirror via mediapipe tasks
"""

import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as tasks_python
from mediapipe.tasks.python import vision as mp_vision

# config cam and display ratios :)
MODEL_PATH = "hand_landmarker.task"
CAM_W, CAM_H = 640, 480
CANVAS_W, CANVAS_H = 640, 480
MAX_HANDS = 3


def normalized_to_pixel(nx, ny, w, h):
    return int(nx * w), int(ny * h)


def hand_landmarks_to_points(hand_landmarks, w, h):
    pts = []
    for lm in hand_landmarks:
        pts.append(normalized_to_pixel(lm.x, lm.y, w, h))
    return pts


FINGER_CONNECTIONS = [
    (0,1,2,3,4),
    (0,5,6,7,8),
    (0,9,10,11,12),
    (0,13,14,15,16),
    (0,17,18,19,20),
]


def draw_stylized_hand(canvas, pts):
    if len(pts) != 21:
        return

    palm_idx = [0,1,5,9,13,17]
    palm = np.array([pts[i] for i in palm_idx], np.int32)
    cv2.fillPoly(canvas, [palm], (200,200,200))

    for finger in FINGER_CONNECTIONS:
        finger_pts = np.array([pts[i] for i in finger], np.int32)
        cv2.polylines(
            canvas,
            [finger_pts],
            False,
            (50,50,200),
            8,
            cv2.LINE_AA
        )
        for x, y in finger_pts:
            cv2.circle(canvas, (x, y), 6, (0,40,200), -1)

    cx = int(np.mean([p[0] for p in palm]))
    cy = int(np.mean([p[1] for p in palm]))
    cv2.circle(canvas, (cx, cy), 10, (0,120,0), -1)


# sets up tasks, had to switch from solutions since its kinda deprecated </3
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

# camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

if not cap.isOpened():
    raise SystemExit("camera failed")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        ts = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, ts)

        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

        # handle up to MAX_HANDS detected
        if result and result.hand_landmarks:
            for idx, hand in enumerate(result.hand_landmarks[:MAX_HANDS]):
                pts = hand_landmarks_to_points(hand, CANVAS_W, CANVAS_H)
                draw_stylized_hand(canvas, pts)

                # dots showing debug areas (displays as green dots on hand)
                # use different debug color per hand for clarity
                dbg_color = (0,255,0) if idx == 0 else (0,180,255)
                for lm in hand:
                    x, y = normalized_to_pixel(lm.x, lm.y, CAM_W, CAM_H)
                    cv2.circle(frame, (x, y), 4, dbg_color, -1)

        out = np.hstack((
            cv2.resize(frame, (CAM_W, CAM_H)),
            cv2.resize(canvas, (CANVAS_W, CANVAS_H))
        ))

        cv2.putText(out, "Camera", (20,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(out, "Mirrored Hand", (CAM_W+20,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.imshow("hand mirror", out)

        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    landmarker.__del__()