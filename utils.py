import cv2
import mediapipe as mp
import numpy as np

def initialize_mediapipe():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mp_drawing = mp.solutions.drawing_utils
    return hands, mp_drawing

def process_frame(frame, hands):
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Extract hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get bounding box for cropping
            h, w, _ = frame.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

            # Add padding to avoid cropping issues
            padding = 20
            x_min = max(0, x_min - padding)
            x_max = min(w, x_max + padding)
            y_min = max(0, y_min - padding)
            y_max = min(h, y_max + padding)

            # Crop and resize hand region
            hand_crop = frame[y_min:y_max, x_min:x_max]
            if hand_crop.size > 0:
                hand_crop = cv2.resize(hand_crop, (64, 64))
                return hand_crop, hand_landmarks
    return None, None

def draw_landmarks(frame, hand_landmarks, mp_drawing):
    if hand_landmarks:
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
    return frame