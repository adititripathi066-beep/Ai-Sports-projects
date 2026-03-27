import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

st.title("🏏 Cricket AI Analyzer PRO")

# 👉 Angle function
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle

# 👉 Mediapipe setup
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# 👉 Streamlit placeholders
video_placeholder = st.empty()
counter_placeholder = st.empty()
accuracy_placeholder = st.empty()
status_placeholder = st.empty()

# 👉 Counter variables
count = 0
stage = None
correct = 0

# 👉 Start camera
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        st.warning("⚠️ Camera not found or not accessible")
        break

    frame = cv2.flip(frame, 1)  # Mirror image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark
        shoulder = [landmarks[12].x, landmarks[12].y]
        elbow = [landmarks[14].x, landmarks[14].y]
        wrist = [landmarks[16].x, landmarks[16].y]

        angle = calculate_angle(shoulder, elbow, wrist)

        # 👉 AI Logic
        if angle > 160:
            result_text = "Good Bowling Action ✅"
            color = (0, 255, 0)
            stage = "up"
        else:
            result_text = "Wrong Action ❌"
            color = (0, 0, 255)

        # 👉 Counting logic
        if angle < 140 and stage == "up":
            stage = "down"
            count += 1
            if angle > 160:
                correct += 1

        accuracy = (correct / count) * 100 if count > 0 else 0

        # 👉 Draw skeleton
        mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 👉 Overlay info on frame
        cv2.putText(frame, f"Angle: {int(angle)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, result_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Balls: {count}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
        cv2.putText(frame, f"Correct: {correct}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.putText(frame, f"Accuracy: {int(accuracy)}%", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)

    # 👉 Streamlit display
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_placeholder.image(frame, channels="RGB")
    counter_placeholder.text(f"Total Balls: {count} | Correct: {correct}")
    accuracy_placeholder.text(f"Accuracy: {int(accuracy)}%")
    status_placeholder.text(f"Status: {result_text}")

# 👉 Release resources
cap.release()