import os
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from datetime import datetime

# ---------------- YOLO 模型載入 ---------------- #
yolo_pose_model = YOLO("yolov8n.pt")
yolo_seg_model = YOLO("yolov8s-seg.pt")
yolo_emt_model = YOLO("yolov8n.pt")
target_classes = ["EMT", "EMTLOGO"]

# ---------------- Mediapipe ---------------- #
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_model = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.7, model_complexity=2)

# ---------------- MongoDB 初始化 ---------------- #
client = MongoClient("mongodb://localhost:27017/")
try:
    client.admin.command('ping')
    print("✅ MongoDB 連線成功！")
except ConnectionFailure:
    print("❌ MongoDB 連線失敗！")

db = client["elevator_emergency_dbb"]
collection_emergency = db["emergencies"]
collection_units = db["detected_units"]

# ---------------- 通用函式 ---------------- #
def calculate_joint_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def classify_posture(shoulder, hip, knee, height_threshold=40):
    hip_angle = calculate_joint_angle(shoulder, hip, knee)
    height_diff = abs(shoulder[1] - hip[1])
    if hip_angle < 160:
        return "Lying Down" if height_diff < height_threshold or hip_angle < 80 else "Sitting"
    else:
        return "Standing"

def detect_people(frame):
    results = yolo_pose_model(frame)
    persons = []
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0:
                persons.append(tuple(map(int, box.xyxy[0])))
    return persons

def detect_pose(roi):
    frame_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    results = pose_model.process(frame_rgb)
    return results.pose_landmarks if results.pose_landmarks else None, results

def analyze_segmentation(frame):
    results = yolo_seg_model(frame, task="segment")
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for result in results:
        if result.masks is not None:
            for seg, cls_id in zip(result.masks.xyn, result.boxes.cls):
                cls_name = result.names[int(cls_id)]
                if cls_name in ['person', 'backpack', 'suitcase']:
                    seg = np.array(seg * [frame.shape[1], frame.shape[0]], dtype=np.int32)
                    cv2.fillPoly(mask, [seg], 255)
    return cv2.countNonZero(mask) / mask.size

def check_emergency_status(postures):
    return 2 if "Lying Down" in postures else (1 if "Sitting" in postures else 0)

def save_emergency_to_mongodb(filename, emergency_flag, postures, occupancy_ratio):
    record = {
        "timestamp": datetime.now(),
        "filename": filename,
        "emergency_flag": emergency_flag,
        "postures": postures,
        "occupancy_ratio": occupancy_ratio
    }
    collection_emergency.insert_one(record)
    print(f"✅ 緊急事件紀錄寫入 MongoDB 成功（{filename}）")

def save_detected_units_to_mongodb(frame_id, detected_targets):
    record = {
        "timestamp": datetime.now(),
        "frame_id": frame_id,
        "detected_units": list(detected_targets)
    }
    collection_units.insert_one(record)
    print(f"✅ 醫護警消紀錄寫入 MongoDB 成功（frame: {frame_id}）")

# ---------------- 模式選擇 ---------------- #
MODE = input("選擇模式：1 = 圖片分析, 2 = EMT Webcam 傳印健檢：")

if MODE == "1":
    IMAGE_FOLDER = "images"
    OUTPUT_FOLDER = "output"
    THRESHOLD = 0.30
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    for filename in sorted(image_files):
        path = os.path.join(IMAGE_FOLDER, filename)
        image = cv2.imread(path)
        persons = detect_people(image)
        postures = []

        for (x1, y1, x2, y2) in persons:
            roi = image[y1:y2, x1:x2]
            landmarks, results = detect_pose(roi)
            color = (0, 255, 0)
            msgs = []

            if landmarks:
                shoulder = (landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * roi.shape[1],
                            landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * roi.shape[0])
                hip = (landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * roi.shape[1],
                       landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * roi.shape[0])
                knee = (landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x * roi.shape[1],
                        landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * roi.shape[0])
                angle = calculate_joint_angle(shoulder, hip, knee)
                posture = classify_posture(shoulder, hip, knee)
                color = (255, 0, 0) if posture == "Lying Down" else (0, 165, 255) if posture == "Sitting" else (0, 255, 0)
                msgs.append(f"{posture} Angle: {angle:.1f}")
                postures.append(posture)
                mp_drawing.draw_landmarks(roi, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            for i, msg in enumerate(msgs):
                cv2.putText(image, msg, (x1, y1 - 10 - i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        ratio = analyze_segmentation(image)
        status_text = f"Occupancy: {ratio:.1%} - {'FULL' if ratio > THRESHOLD else 'NOT FULL'}"
        status_color = (0, 0, 255) if ratio > THRESHOLD else (0, 255, 0)
        cv2.putText(image, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        emergency_flag = check_emergency_status(postures)
        save_emergency_to_mongodb(filename, emergency_flag, postures, ratio)

        results = yolo_emt_model(image)
        detected_targets = set()
        for result in results:
            for box in result.boxes:
                conf = box.conf[0].item()
                if conf < 0.6:
                    continue
                cls_name = result.names[int(box.cls[0])]
                if cls_name in target_classes:
                    detected_targets.add(cls_name)

        if detected_targets:
            save_detected_units_to_mongodb(filename, detected_targets)

        cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"result_{filename}"), image)
        print(f"📸 分析完成：{filename}\n")

    print("✅ 所有圖片分析完畢")
