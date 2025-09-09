import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import os

# ---------------- 初始化 ---------------- #
yolo_model = YOLO("yolov8n.pt")
clf_model = load_model("MyMobileNetV2_best.h5")
IMG_SIZE = (224, 224)

# ---------------- 函數 ---------------- #
def resize_with_padding(img, size=(224,224)):
    h, w = img.shape[:2]
    scale = min(size[0]/h, size[1]/w)
    nh, nw = int(h*scale), int(w*scale)
    resized = cv2.resize(img, (nw, nh))
    top = (size[0]-nh)//2
    bottom = size[0]-nh-top
    left = (size[1]-nw)//2
    right = size[1]-nw-left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
    return padded

def detect_people(frame):
    results = yolo_model(frame)
    persons = []
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0:  # 只保留人
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                persons.append((x1, y1, x2, y2))
    return persons

def classify_posture_batch(rois, clf_model, IMG_SIZE=(224,224)):
    imgs = np.array([resize_with_padding(roi, IMG_SIZE)/255.0 for roi in rois])
    preds = clf_model.predict(imgs, verbose=0)  # shape: (N,2)
    labels = []
    is_lying_list = []
    for pred in preds:
        lying_prob = pred[1]*100
        if lying_prob < 40:
            is_lying = True
            label_text = f"Lying: {lying_prob:.1f}%"
        else:
            is_lying = False
            label_text = f"Standing: {lying_prob:.1f}%"
        labels.append(label_text)
        is_lying_list.append(is_lying)
    return labels, is_lying_list

def check_emergency(is_lying_list):
    return any(is_lying_list)

# ---------------- 主程式 ---------------- #
video_path = "videos/1.mp4"
output_path = os.path.join("output", "fast_analyzed_batch.mp4")
os.makedirs("output", exist_ok=True)

cap = cv2.VideoCapture(video_path)
width, height = int(cap.get(3)), int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    persons = detect_people(frame)
    rois = [frame[y1:y2, x1:x2] for (x1, y1, x2, y2) in persons]

    if rois:
        labels, is_lying_list = classify_posture_batch(rois, clf_model)

        for (x1, y1, x2, y2), label_text, is_lying in zip(persons, labels, is_lying_list):
            color = (0,0,255) if is_lying else (0,255,0)  # 紅框 Lying，綠框 Standing
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label_text, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if check_emergency(is_lying_list):
            print(f"⚠️ Frame {frame_count}: Emergency detected!")

    out.write(frame)
    frame_count += 1

cap.release()
out.release()
print(f"🎉 分析完成，影片輸出到：{output_path}")
