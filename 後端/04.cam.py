from ultralytics import YOLO
import cv2

model = YOLO("C:/Users/raymo/PycharmProjects/yoloV8/models/best.pt")

target_classes = ["EMT", "EMTLOGO"] 

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("無法開啟攝影機")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("無法讀取畫面")
        break

    # 推論
    results = model(frame)

    detected_targets = set()  # 用來記錄已偵測的目標物件

    for result in results:
        for box in result.boxes:
            conf = box.conf[0].item()
            if conf < 0.6:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = result.names[cls_id]

            if label in target_classes:
                detected_targets.add(label)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if detected_targets:
        print(f"偵測到：{', '.join(detected_targets)}")

    cv2.imshow("Webcam Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
