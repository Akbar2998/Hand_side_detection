import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Compute palm vs dorsum, adjusting for handedness
def compute_palm_side(landmarks, img_w, img_h, handedness):
    # Convert normalized landmarks to pixel 3D coords
    pts = np.array([[lm.x * img_w, lm.y * img_h, lm.z * img_w] for lm in landmarks])
    a, b, c = pts[0], pts[5], pts[17]
    v1 = b - a
    v2 = c - a
    normal = np.cross(v1, v2)
    # Invert normal for right hand to correct orientation
    sign = 1 if handedness == 'Left' else -1
    return normal[2] * sign

# Start capture
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror image for intuitive handedness
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        # Convert to RGB for processing
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # If any hands detected
        if results.multi_hand_landmarks:
            for hand_landmarks, hand_handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                # Draw skeleton
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                # Get handedness label
                label = hand_handedness.classification[0].label  # 'Left' or 'Right'
                # Compute adjusted palm normal
                z_val = compute_palm_side(hand_landmarks.landmark, w, h, label)
                side = 'Palm' if z_val < 0 else 'Dorsum'
                text = f"{label} {side}"

                # Position text at wrist
                wrist = hand_landmarks.landmark[0]
                cx, cy = int(wrist.x * w), int(wrist.y * h)
                cv2.putText(
                    frame,
                    text,
                    (cx - 50, cy - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

        # Display
        cv2.imshow('Hand Side Detection', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

cap.release()
cv2.destroyAllWindows()