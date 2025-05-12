import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

st.set_page_config(page_title="Hand Side Detector", layout="centered")
st.title("Palm vs. Dorsum Detector")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Compute palm vs dorsum, adjusting for handedness
def compute_palm_side(landmarks, img_w, img_h, handedness):
    pts = np.array([[lm.x * img_w, lm.y * img_h, lm.z * img_w] for lm in landmarks])
    a, b, c = pts[0], pts[5], pts[17]
    v1 = b - a
    v2 = c - a
    normal = np.cross(v1, v2)
    # Flip normal for right hand orientation
    sign = 1 if handedness == 'Left' else -1
    return normal[2] * sign

uploaded = st.file_uploader("Upload an image (jpg, jpeg, png)", type=['jpg', 'jpeg', 'png'])
if uploaded:
    image = Image.open(uploaded).convert('RGB')
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w, _ = img.shape

    # Prepare MediaPipe
    with mp_hands.Hands(static_image_mode=True,
                       max_num_hands=2,
                       min_detection_confidence=0.7) as hands:
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, hand_handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            # Draw landmarks
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Raw label from MediaPipe
            raw_label = hand_handedness.classification[0].label  # 'Left' or 'Right'
            # Swap for display so left/right match input orientation
            display_label = 'Right' if raw_label == 'Left' else 'Left'

            # Compute side
            z_val = compute_palm_side(hand_landmarks.landmark, w, h, raw_label)
            side = 'Palm' if z_val < 0 else 'Dorsum'
            text = f"{display_label} {side}"

            # Position at wrist
            wrist = hand_landmarks.landmark[0]
            cx, cy = int(wrist.x * w), int(wrist.y * h)
            cv2.putText(img, text, (cx - 50, cy - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        st.warning("No hands detected. Please try a different image.")

    # Show result
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img, caption="Detection Result", use_column_width=True)
