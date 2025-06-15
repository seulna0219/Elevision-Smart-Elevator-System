import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import math
import os
from socketIO_client import SocketIO

# ---------------- 初始化模型 ---------------- #
socket = SocketIO("127.0.0.1", 5000)  # 改成實際伺服器 IP 與 Port
yolo_model = YOLO("yolov8n.pt")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_model = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, model_complexity=2)

def calculate_joint_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def classify_posture(shoulder, hip, knee, height_threshold=40):
    hip_angle = calculate_joint_angle(shoulder, hip, knee)
    height_diff = abs(shoulder[1] - hip[1])
    if hip_angle < 160:
        if height_diff < height_threshold or hip_angle < 80:
            return "Lying Down"
        else:
            return "Sitting"
    else:
        return "Standing"

def detect_people(frame):
    results = yolo_model(frame)
    persons = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                persons.append((x1, y1, x2, y2))
    return persons

def detectPose(frame, pose_model):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_model.process(frame_rgb)
    return results.pose_landmarks if results.pose_landmarks else None, results

def check_emergency_status(postures):
    for p in postures:
        if p == "Lying Down":
            return 2
        elif p == "Sitting":
            return 1
    return 0

# ---------------- 開啟鏡頭 ---------------- #
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("無法開啟鏡頭")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    persons = detect_people(frame)
    frame_postures = []

    for idx, (x1, y1, x2, y2) in enumerate(persons):
        person_roi = frame[y1:y2, x1:x2]
        landmarks, results = detectPose(person_roi, pose_model)
        warning_messages = []
        color = (0, 255, 0)

        if landmarks:
            shoulder = (landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * person_roi.shape[1],
                        landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * person_roi.shape[0])
            hip = (landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * person_roi.shape[1],
                   landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * person_roi.shape[0])
            knee = (landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x * person_roi.shape[1],
                    landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * person_roi.shape[0])

            angle = calculate_joint_angle(shoulder, hip, knee)
            posture = classify_posture(shoulder, hip, knee)

            if posture == "Lying Down":
                warning_messages.append(f"Lying Down! Angle: {angle:.2f}")
                color = (255, 0, 0)
            elif posture == "Sitting":
                warning_messages.append(f"Sitting! Angle: {angle:.2f}")
                color = (0, 165, 255)

            frame_postures.append(posture)
            mp_drawing.draw_landmarks(frame[y1:y2, x1:x2], results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        for i, msg in enumerate(warning_messages):
            cv2.putText(frame, f"Person {idx+1}: {msg}", (x1, y1 - 10 - (i * 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    emergency_flag = check_emergency_status(frame_postures)
    if emergency_flag in [1, 2]:
        socket.emit("emergency_status", {"level": emergency_flag})

    cv2.putText(frame, f"Emergency: {emergency_flag}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Camera View - Fall Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
